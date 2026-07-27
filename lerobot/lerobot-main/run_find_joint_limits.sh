#!/usr/bin/env bash
# 在 lerobot-main 目录下执行，或使用下方绝对路径的 urdf_path

cd "$(dirname "$0")"

lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower_black \
  --robot.id=black \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader_blue \
  --teleop.id=blue \
  --urdf_path="$(pwd)/SO-ARM100-main/Simulation/SO101/so101_new_calib.urdf" \
  --target_frame_name=gripper \
  --teleop_time_s=30 \
  --warmup_time_s=5 \
  --control_loop_fps=30
