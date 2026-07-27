#!/usr/bin/env python

import logging
import os
import select
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so_leader_spacebar_follow import SOLeaderSpacebarFollowTeleopConfig

logger = logging.getLogger(__name__)

# Default step sizes (meters per unit action) when converting leader ee delta to action
DEFAULT_STEP_SIZES = {"x": 0.02, "y": 0.02, "z": 0.02}


class SOLeaderSpacebarFollowTeleop(Teleoperator):
    """Independent teleop: leader follows follower, space toggles intervention."""

    config_class = SOLeaderSpacebarFollowTeleopConfig
    name = "so_leader_spacebar_follow"

    def __init__(self, config: SOLeaderSpacebarFollowTeleopConfig):
        super().__init__(config)
        self.config = config
        # Leader bus needs calibration for sync_write; reuse so101_leader calibration if needed
        if getattr(config, "leader_calibration_path", None):
            p = Path(config.leader_calibration_path)
            if p.is_file():
                self._load_calibration(p)
        if not self.calibration:
            for candidate in [
                self.calibration_dir.parent / "so_leader" / "None.json",
                self.calibration_dir.parent / "so_leader" / "so101_leader.json",
            ]:
                if candidate.is_file():
                    self._load_calibration(candidate)
                    logger.info("Using leader calibration from %s", candidate)
                    break

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        self.leader_bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.follower_bus = FeetechMotorsBus(
            port=self.config.follower_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )

        self._stdin_fd: int | None = None
        self._stdin_old_settings = None
        self._stdin_mode = False
        self._stdin_owns_fd = False
        self._misc_keys_queue: Queue[str] = Queue()

        self._intervention_enabled = False
        self._terminate_episode = False
        self._rerecord_episode = False

        self._robot = None  # Follower robot, set by env via set_robot()
        self._kinematics = None
        self._motor_names = list(self.leader_bus.motors.keys())
        if config.urdf_path:
            try:
                from lerobot.model.kinematics import RobotKinematics

                self._kinematics = RobotKinematics(
                    urdf_path=config.urdf_path,
                    target_frame_name=config.target_frame_name,
                    joint_names=self._motor_names,
                )
            except Exception as e:
                logger.warning(f"Could not create kinematics for leader-follow intervention: {e}")
        self._step_sizes = config.end_effector_step_sizes or DEFAULT_STEP_SIZES

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.leader_bus.is_connected and self.follower_bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader_bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.leader_bus.connect()
        self.follower_bus.connect()
        try:
            if sys.stdin.isatty():
                self._setup_stdin(sys.stdin.fileno(), owns_fd=False)
            else:
                fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
                self._setup_stdin(fd, owns_fd=True)
        except Exception as e:
            logger.warning(f"Keyboard event capture unavailable ({e}). Spacebar intervention disabled.")

        logger.info(
            "SO leader spacebar follow connected. "
            "SPACE=toggle intervention (reward=1), s=success, q=fail, r=rerecord."
        )

    def _setup_stdin(self, fd: int, owns_fd: bool) -> None:
        import termios
        import tty

        self._stdin_fd = fd
        self._stdin_old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._stdin_mode = True
        self._stdin_owns_fd = owns_fd

    def set_robot(self, robot: Any) -> None:
        """Set the follower robot reference so intervention can read current state and output ee delta."""
        self._robot = robot

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _drain_key_events(self) -> None:
        if not self._stdin_mode or self._stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0)
            if not ready:
                break
            raw = os.read(self._stdin_fd, 1)
            if not raw:
                break
            key = raw.decode("utf-8", errors="ignore")
            if key == " ":
                self._intervention_enabled = not self._intervention_enabled
                # 介入时关闭主臂力矩便于轻松拖动，退出介入时恢复力矩以便 mirror 控制
                try:
                    if self._intervention_enabled:
                        self.leader_bus.disable_torque(num_retry=1)
                        logger.info("Leader torque OFF → easy to drag (intervention on)")
                    else:
                        self.leader_bus.enable_torque(num_retry=1)
                        logger.info("Leader torque ON (intervention off)")
                except Exception as e:
                    logger.warning("Leader torque toggle failed: %s", e)
                logger.info(
                    "Space toggled intervention=%s (%s)",
                    self._intervention_enabled,
                    "human in control" if self._intervention_enabled else "policy in control",
                )
            elif key in {"s", "q", "r"}:
                self._misc_keys_queue.put(key)
                if key == "s":
                    logger.info("Key 's' (success) queued → episode will end with reward=1")
                elif key == "q":
                    logger.info("Key 'q' queued → episode will end")
                elif key == "r":
                    logger.info("Key 'r' queued → episode will end and rerecord")

    def _get_follower_bus_for_read(self):
        """Use robot's bus (has calibration) when set, else our follower_bus."""
        if self._robot is not None and getattr(self._robot, "bus", None) is not None:
            return self._robot.bus
        return self.follower_bus

    def _mirror_follower_to_leader(self) -> None:
        if not self.config.mirror_follower or self._intervention_enabled:
            return
        bus = self._get_follower_bus_for_read()
        follower_q = bus.sync_read("Present_Position", num_retry=1)
        self.leader_bus.sync_write("Goal_Position", follower_q, num_retry=1)

    def _leader_follow_current_position(self, repeat: int = 1) -> None:
        """During intervention: set leader goal = leader present so the arm is transparent.
        When repeat > 1, write multiple times with short sleep to make leader feel freer (like record mode)."""
        if not self._intervention_enabled:
            return
        try:
            for _ in range(max(1, repeat)):
                leader_q = self.leader_bus.sync_read("Present_Position", num_retry=1)
                self.leader_bus.sync_write("Goal_Position", leader_q, num_retry=1)
                if repeat > 1:
                    time.sleep(0.012)
        except Exception as e:
            logger.debug("Leader follow-current failed: %s", e)

    def reset_for_new_episode(self) -> None:
        """新回合开始前调用：退出介入、恢复主臂力矩，下一轮由策略直接推理控制。"""
        if self._intervention_enabled:
            self._intervention_enabled = False
            try:
                self.leader_bus.enable_torque(num_retry=1)
                logger.info("New episode: intervention cleared, leader torque ON (policy in control)")
            except Exception as e:
                logger.warning("reset_for_new_episode torque enable failed: %s", e)

    def notify_reset(
        self,
        target_joint_positions: list[float] | None,
        duration_sec: float = 0.0,
    ) -> None:
        """On round end: leader actively resets to the given pose by itself (same pose as follower).

        When duration_sec > 0, drives the leader to the pose for that many seconds so it resets
        in parallel with the follower. Temporarily enables torque to move, then restores torque
        state (off if still in intervention so user can drag again next episode).
        """
        if target_joint_positions is None:
            return
        n = min(len(self._motor_names), len(target_joint_positions))
        if n == 0:
            return
        try:
            # 复位时需要力矩才能移动主臂
            self.leader_bus.enable_torque(num_retry=1)
            q_target = {self._motor_names[i]: float(target_joint_positions[i]) for i in range(n)}
            if duration_sec <= 0:
                self.leader_bus.sync_write("Goal_Position", q_target, num_retry=1)
                logger.info("Leader synced to reset pose (%d joints)", n)
            else:
                # 平滑轨迹：从当前位插值到目标，duration_sec 内缓慢归位，避免冲击
                current = self.leader_bus.sync_read("Present_Position", num_retry=1)
                pos_start = np.array([float(current[self._motor_names[i]]) for i in range(n)], dtype=np.float64)
                pos_end = np.array([float(target_joint_positions[i]) for i in range(n)], dtype=np.float64)
                num_steps = max(50, int(duration_sec / 0.03))
                step_sleep = duration_sec / num_steps
                for i in range(num_steps + 1):
                    t = i / num_steps
                    pos = pos_start + t * (pos_end - pos_start)
                    q = {self._motor_names[j]: float(pos[j]) for j in range(n)}
                    self.leader_bus.sync_write("Goal_Position", q, num_retry=1)
                    time.sleep(step_sleep)
                logger.info("Leader reset to pose over %.1fs (%d joints)", duration_sec, n)
            # 若仍处于介入，恢复无力矩以便下一轮轻松拖动
            if self._intervention_enabled:
                self.leader_bus.disable_torque(num_retry=1)
        except Exception as e:
            logger.warning("notify_reset: could not sync leader to reset pose: %s", e)

    def get_leader_joint_positions(self) -> list[float] | None:
        """Return current leader joint positions in motor order (same as follower), for joint mirroring during intervention."""
        try:
            leader_pos = self.leader_bus.sync_read("Present_Position", num_retry=3)
            return [float(leader_pos[m]) for m in self._motor_names]
        except Exception as e:
            if self._intervention_enabled:
                logger.debug("get_leader_joint_positions failed: %s", e)
            return None

    def _leader_action_as_ee_delta(self) -> dict[str, float] | None:
        """Convert current leader and follower joints to ee delta action. Returns None if unavailable."""
        if self._robot is None or self._kinematics is None:
            if self._intervention_enabled:
                logger.warning("Intervention on but leader->ee delta unavailable (robot or kinematics missing)")
            return None
        bus = self._get_follower_bus_for_read()
        try:
            leader_pos = self.leader_bus.sync_read("Present_Position", num_retry=3)
            follower_pos = bus.sync_read("Present_Position", num_retry=3)
        except Exception as e:
            if self._intervention_enabled:
                logger.warning("Intervention on but sync_read failed: %s", e)
            return None
        leader_j = np.array([float(leader_pos[m]) for m in self._motor_names], dtype=np.float64)
        follower_j = np.array([float(follower_pos[m]) for m in self._motor_names], dtype=np.float64)
        T_leader = self._kinematics.forward_kinematics(leader_j)
        T_follower = self._kinematics.forward_kinematics(follower_j)
        p_leader = T_leader[:3, 3]
        p_follower = T_follower[:3, 3]
        delta_m = p_leader - p_follower
        sx = self._step_sizes.get("x", DEFAULT_STEP_SIZES["x"])
        sy = self._step_sizes.get("y", DEFAULT_STEP_SIZES["y"])
        sz = self._step_sizes.get("z", DEFAULT_STEP_SIZES["z"])
        # Action units: delta_meters / step_size, clip to avoid huge steps
        delta_x = np.clip(delta_m[0] / sx, -2.0, 2.0)
        delta_y = np.clip(delta_m[1] / sy, -2.0, 2.0)
        delta_z = np.clip(delta_m[2] / sz, -2.0, 2.0)
        gripper = float(leader_pos.get("gripper", 50.0)) / 100.0
        action = {"delta_x": float(delta_x), "delta_y": float(delta_y), "delta_z": float(delta_z)}
        if self.config.use_gripper:
            action["gripper"] = float(np.clip(gripper, 0.0, 1.0))
        return action

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        self._drain_key_events()
        if not self._intervention_enabled:
            self._mirror_follower_to_leader()
        # 介入时已 disable_torque，不再写 Goal_Position，避免主臂有保持力导致手感发紧
        if self._intervention_enabled:
            ee_action = self._leader_action_as_ee_delta()
            if ee_action is not None:
                return ee_action
        action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        if self.config.use_gripper:
            action["gripper"] = 1.0
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        self._drain_key_events()

        terminate_episode = False
        rerecord_episode = False
        success_from_key = False
        while not self._misc_keys_queue.empty():
            key = self._misc_keys_queue.get_nowait()
            if key == "s":
                terminate_episode = True
                success_from_key = True
            elif key == "q":
                terminate_episode = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_enabled,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success_from_key,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._stdin_mode and self._stdin_fd is not None and self._stdin_old_settings is not None:
            try:
                import termios

                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:
                pass
        if self._stdin_owns_fd and self._stdin_fd is not None:
            try:
                os.close(self._stdin_fd)
            except Exception:
                pass
        self._stdin_mode = False
        self._stdin_owns_fd = False

        self.follower_bus.disconnect()
        self.leader_bus.disconnect()
        logger.info(f"{self} disconnected.")
