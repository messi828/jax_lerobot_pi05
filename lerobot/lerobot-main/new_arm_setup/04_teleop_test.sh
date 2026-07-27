#!/usr/bin/env bash
# 遥操作测试：验证新主从臂校准是否正确（不录数据）
# 移动主臂，从臂应平滑跟随；Ctrl+C 退出

source /home/jaylen/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_rl

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower2 \
  --robot.id=follower2 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader2 \
  --teleop.id=leader2
