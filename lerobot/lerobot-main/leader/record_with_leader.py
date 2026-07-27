#!/usr/bin/env python3
"""
Record demonstrations with SO101 leader -> follower direct control.

This script avoids gym_manipulator teleop-events pipeline and records a LeRobotDataset
by directly:
1) reading leader joint action
2) sending action to follower
3) saving follower observations + action

Episode label is provided manually at the end of each episode:
  s = success, f = failure, r = rerecord, q = quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.cameras import opencv  # noqa: F401
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.teleoperators import make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class LeaderRecordConfig:
    fps: int
    control_time_s: float
    reset_time_s: float
    fixed_reset_joint_positions: list[float] | None
    num_episodes_to_record: int
    task: str
    repo_id: str
    root: str | None
    push_to_hub: bool

    follower_port: str
    follower_use_degrees: bool
    follower_id: str | None
    leader_port: str
    leader_use_degrees: bool
    leader_id: str | None
    cameras: dict[str, dict[str, Any]]

    @staticmethod
    def from_json(path: str) -> "LeaderRecordConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        return LeaderRecordConfig(
            fps=int(raw["record"]["fps"]),
            control_time_s=float(raw["record"]["control_time_s"]),
            reset_time_s=float(raw["record"]["reset_time_s"]),
            fixed_reset_joint_positions=raw["record"].get("fixed_reset_joint_positions"),
            num_episodes_to_record=int(raw["dataset"]["num_episodes_to_record"]),
            task=str(raw["dataset"]["task"]),
            repo_id=str(raw["dataset"]["repo_id"]),
            root=raw["dataset"].get("root"),
            push_to_hub=bool(raw["dataset"].get("push_to_hub", False)),
            follower_port=str(raw["follower"]["port"]),
            follower_use_degrees=bool(raw["follower"].get("use_degrees", True)),
            follower_id=raw["follower"].get("id"),
            leader_port=str(raw["leader"]["port"]),
            leader_use_degrees=bool(raw["leader"].get("use_degrees", True)),
            leader_id=raw["leader"].get("id"),
            cameras=dict(raw["follower"]["cameras"]),
        )


def make_camera_config(cam: dict[str, Any]) -> CameraConfig:
    cam_type = str(cam.get("type", "opencv")).lower()
    fps = int(cam.get("fps", 30))
    width = int(cam.get("width", 640))
    height = int(cam.get("height", 480))
    if cam_type in {"intelrealsense", "realsense"}:
        return RealSenseCameraConfig(
            serial_number_or_name=str(cam["serial_number_or_name"]),
            fps=fps,
            width=width,
            height=height,
            use_depth=bool(cam.get("use_depth", False)),
        )
    return OpenCVCameraConfig(
        index_or_path=cam["index_or_path"],
        fps=fps,
        width=width,
        height=height,
    )


def make_follower(cfg: LeaderRecordConfig):
    camera_cfgs: dict[str, CameraConfig] = {
        name: make_camera_config(cam) for name, cam in cfg.cameras.items()
    }

    follower_cfg = SOFollowerRobotConfig(
        port=cfg.follower_port,
        use_degrees=cfg.follower_use_degrees,
        cameras=camera_cfgs,
        id=cfg.follower_id,
    )
    return make_robot_from_config(follower_cfg)


def make_leader(cfg: LeaderRecordConfig):
    leader_cfg = SOLeaderTeleopConfig(
        port=cfg.leader_port,
        use_degrees=cfg.leader_use_degrees,
        id=cfg.leader_id,
    )
    return make_teleoperator_from_config(leader_cfg)


def reset_follower_position(robot, target_position: list[float]) -> None:
    current_position_dict = robot.bus.sync_read("Present_Position")
    motor_names = list(current_position_dict.keys())
    current_position = np.array([current_position_dict[name] for name in motor_names], dtype=np.float32)
    target = np.array(target_position, dtype=np.float32)
    traj = np.linspace(current_position, target, 50)
    for pose in traj:
        action = {f"{name}.pos": float(val) for name, val in zip(motor_names, pose, strict=True)}
        robot.send_action(action)
        precise_sleep(0.015)


def to_action_vector(action_dict: dict[str, float], motor_names: list[str]) -> np.ndarray:
    vals = [float(action_dict[f"{m}.pos"]) for m in motor_names]
    return np.asarray(vals, dtype=np.float32)


def to_state_vector(obs_dict: dict[str, Any], motor_names: list[str]) -> np.ndarray:
    vals = [float(obs_dict[f"{m}.pos"]) for m in motor_names]
    return np.asarray(vals, dtype=np.float32)


def to_image_tensor_hwc_to_chw(img: np.ndarray) -> np.ndarray:
    # img from OpenCV camera class in lerobot is RGB uint8 HWC.
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    return np.transpose(img, (2, 0, 1)).astype(np.uint8)


def ask_episode_label(ep_idx: int, total: int) -> str:
    while True:
        cmd = input(
            f"[Episode {ep_idx + 1}/{total}] label: [s]uccess / [f]ailure / [r]erecord / [q]uit > "
        ).strip().lower()
        if cmd in {"s", "f", "r", "q"}:
            return cmd
        print("Invalid input. Use s/f/r/q.")


def build_features(first_obs: dict[str, Any], motor_names: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        ACTION: {"dtype": "float32", "shape": (len(motor_names),), "names": None},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (len(motor_names),), "names": None},
    }
    for key, value in first_obs.items():
        if key.endswith(".pos"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 3:
            chw = to_image_tensor_hwc_to_chw(value)
            features[f"{OBS_IMAGES}.{key}"] = {
                "dtype": "video",
                "shape": chw.shape,
                "names": ["channels", "height", "width"],
            }
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(REPO_ROOT / "leader" / "leader_record_config.json"),
        help="Path to leader record config json",
    )
    args = parser.parse_args()

    cfg = LeaderRecordConfig.from_json(args.config_path)
    follower = make_follower(cfg)
    leader = make_leader(cfg)

    print("Connecting follower and leader...")
    follower.connect()
    leader.connect()
    print("Connected.")

    motor_names = list(follower.bus.motors.keys())
    print("Follower motors:", motor_names)

    first_obs = follower.get_observation()
    features = build_features(first_obs, motor_names)

    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    dt = 1.0 / cfg.fps
    episode_idx = 0
    print(f"Start recording. episodes={cfg.num_episodes_to_record}, fps={cfg.fps}, control_time_s={cfg.control_time_s}")

    try:
        while episode_idx < cfg.num_episodes_to_record:
            if cfg.fixed_reset_joint_positions is not None:
                print("Resetting follower pose...")
                reset_follower_position(follower, cfg.fixed_reset_joint_positions)
            precise_sleep(cfg.reset_time_s)

            print(
                f"[Episode {episode_idx + 1}/{cfg.num_episodes_to_record}] "
                f"recording for up to {cfg.control_time_s:.1f}s ..."
            )
            episode_start = time.perf_counter()
            last_log_s = -1
            step_count = 0
            while (time.perf_counter() - episode_start) < cfg.control_time_s:
                step_t0 = time.perf_counter()

                leader_action = leader.get_action()
                follower.send_action(leader_action)
                obs = follower.get_observation()

                frame = {
                    OBS_STATE: to_state_vector(obs, motor_names),
                    ACTION: to_action_vector(leader_action, motor_names),
                    REWARD: np.array([0.0], dtype=np.float32),
                    DONE: np.array([False], dtype=bool),
                    "task": cfg.task,
                }

                for key, value in obs.items():
                    if key.endswith(".pos"):
                        continue
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        frame[f"{OBS_IMAGES}.{key}"] = to_image_tensor_hwc_to_chw(value)

                dataset.add_frame(frame)
                step_count += 1

                elapsed_s = int(time.perf_counter() - episode_start)
                if elapsed_s != last_log_s:
                    last_log_s = elapsed_s
                    remaining = max(int(cfg.control_time_s) - elapsed_s, 0)
                    print(
                        f"  recording... elapsed={elapsed_s}s remaining={remaining}s frames={step_count}",
                        end="\r",
                        flush=True,
                    )
                precise_sleep(max(dt - (time.perf_counter() - step_t0), 0.0))

            print()
            cmd = ask_episode_label(episode_idx, cfg.num_episodes_to_record)
            if cmd == "q":
                print("Quit requested. Clearing current episode buffer.")
                dataset.clear_episode_buffer()
                break
            if cmd == "r":
                print("Re-record requested. Clearing current episode buffer.")
                dataset.clear_episode_buffer()
                continue

            success = cmd == "s"

            # Add one terminal frame carrying final reward/done label.
            leader_action = leader.get_action()
            obs = follower.get_observation()
            terminal = {
                OBS_STATE: to_state_vector(obs, motor_names),
                ACTION: to_action_vector(leader_action, motor_names),
                REWARD: np.array([1.0 if success else 0.0], dtype=np.float32),
                DONE: np.array([True], dtype=bool),
                "task": cfg.task,
            }
            for key, value in obs.items():
                if key.endswith(".pos"):
                    continue
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    terminal[f"{OBS_IMAGES}.{key}"] = to_image_tensor_hwc_to_chw(value)
            dataset.add_frame(terminal)
            dataset.save_episode()
            episode_idx += 1
            print(f"Saved episode {episode_idx} / {cfg.num_episodes_to_record} (success={success})")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping recording loop.")

    finally:
        print("Disconnecting...")
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            follower.disconnect()
        except Exception:
            pass
        if cfg.push_to_hub:
            print("Pushing dataset to hub...")
            dataset.push_to_hub()
        print("Done.")


if __name__ == "__main__":
    main()

