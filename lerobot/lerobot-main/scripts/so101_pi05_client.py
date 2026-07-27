#!/usr/bin/env python3
"""SO101 真机客户端：连接 openpi (JAX pi0.5) WebSocket 策略服务器做闭环推理。

服务器端（JAX/训练环境，openpi 工程根目录）：
    python scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_so101 \
        --policy.dir=checkpoints/pi05_so101/so101_pick/<step>

客户端（lerobot_rl 环境，lerobot-main 目录）：
    python scripts/so101_pi05_client.py --task "pick_lift_cube"

观测/动作格式与训练数据一致（record_with_leader.py 采集格式）：
    observation.state: (6,) 关节角（度，gripper 0-100）
    observation.images.front / wrist: HWC RGB uint8
    动作: (H, 6) 绝对关节目标
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi_client import websocket_client_policy

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.robot_utils import precise_sleep

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO101 + JAX pi0.5 websocket 推理客户端")
    p.add_argument("--host", type=str, default="localhost", help="策略服务器地址")
    p.add_argument("--port", type=int, default=8000, help="策略服务器端口")
    p.add_argument("--task", type=str, default="pick_lift_cube", help="任务指令 prompt")
    p.add_argument("--fps", type=int, default=10, help="控制频率（与采集一致）")
    p.add_argument(
        "--n-execute",
        type=int,
        default=10,
        help="每个 action chunk 实际执行的步数（<= 模型 action_horizon=10），越小越闭环",
    )
    p.add_argument("--robot-port", type=str, default="/dev/so101_follower2", help="从臂串口")
    p.add_argument("--robot-id", type=str, default="follower2", help="从臂校准 ID")
    p.add_argument(
        "--front-camera",
        type=str,
        default="realsense:050522073017",
        help="全局相机：realsense:<serial> 或 opencv:<path>",
    )
    p.add_argument(
        "--wrist-camera",
        type=str,
        default="opencv:/dev/video-so101-wrist",
        help="腕部相机：realsense:<serial> 或 opencv:<path>",
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=30.0,
        help="单步关节最大相对移动量（度），安全限幅；<=0 表示不限制",
    )
    p.add_argument(
        "--reset-joints",
        type=float,
        nargs=6,
        default=None,
        metavar=("PAN", "LIFT", "ELBOW", "WFLEX", "WROLL", "GRIP"),
        help="启动时缓慢移动到该关节位（度）；不传则从当前位姿直接开始",
    )
    p.add_argument("--max-steps", type=int, default=0, help="最大控制步数，0 表示无限直到 Ctrl+C")
    return p.parse_args()


def make_camera_config(spec: str):
    kind, _, value = spec.partition(":")
    if kind == "realsense":
        return RealSenseCameraConfig(
            serial_number_or_name=value, width=640, height=480, fps=30
        )
    if kind == "opencv":
        return OpenCVCameraConfig(index_or_path=value, width=640, height=480, fps=30)
    raise ValueError(f"无法解析相机配置: {spec}（应为 realsense:<serial> 或 opencv:<path>）")


def make_robot(args: argparse.Namespace):
    cfg = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=args.max_relative_target if args.max_relative_target > 0 else None,
        cameras={
            "front": make_camera_config(args.front_camera),
            "wrist": make_camera_config(args.wrist_camera),
        },
    )
    return make_robot_from_config(cfg)


def slow_move_to(robot, target: list[float], duration_s: float = 3.0) -> None:
    """缓慢插值移动到目标关节位，避免上电瞬间猛冲。"""
    current = robot.bus.sync_read("Present_Position")
    start = np.array([current[m] for m in MOTOR_NAMES], dtype=np.float32)
    goal = np.array(target, dtype=np.float32)
    n = max(int(duration_s * 30), 1)
    for pose in np.linspace(start, goal, n):
        robot.send_action({f"{m}.pos": float(v) for m, v in zip(MOTOR_NAMES, pose, strict=True)})
        precise_sleep(duration_s / n)


def build_observation(robot, task: str) -> dict:
    obs = robot.get_observation()
    state = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)
    return {
        "observation.state": state,
        "observation.images.front": np.asarray(obs["front"]),
        "observation.images.wrist": np.asarray(obs["wrist"]),
        "prompt": task,
    }


def main() -> None:
    args = parse_args()
    dt = 1.0 / args.fps

    print(f"连接策略服务器 ws://{args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("服务器元信息:", client.get_server_metadata())

    print("连接 SO101 从臂与相机 ...")
    robot = make_robot(args)
    robot.connect()
    print("已连接。任务:", args.task)

    if args.reset_joints is not None:
        print("缓慢移动到初始位:", args.reset_joints)
        slow_move_to(robot, args.reset_joints)

    step = 0
    try:
        while True:
            obs = build_observation(robot, args.task)

            t0 = time.perf_counter()
            result = client.infer(obs)
            actions = np.asarray(result["actions"])  # (H, 6)
            infer_ms = (time.perf_counter() - t0) * 1000
            if actions.ndim == 1:
                actions = actions[None, :]
            print(
                f"step={step} infer={infer_ms:.0f}ms chunk={actions.shape} "
                f"state={np.round(obs['observation.state'], 1)}"
            )

            for action in actions[: args.n_execute]:
                step_t0 = time.perf_counter()
                robot.send_action(
                    {f"{m}.pos": float(v) for m, v in zip(MOTOR_NAMES, action, strict=True)}
                )
                step += 1
                if args.max_steps and step >= args.max_steps:
                    raise KeyboardInterrupt
                precise_sleep(max(dt - (time.perf_counter() - step_t0), 0.0))

    except KeyboardInterrupt:
        print("\n停止推理。")
    finally:
        print("断开连接 ...")
        try:
            robot.disconnect()
        except Exception:
            pass
        print("完成。")


if __name__ == "__main__":
    main()
