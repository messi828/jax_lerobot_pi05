#!/usr/bin/env bash
# 校准新从臂（SO101 follower，ID: follower2）
# 按提示：先把各关节摆到中位并确认，然后手动缓慢转动每个关节到两端极限，最后回车结束

source /home/jaylen/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_rl

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower2 \
  --robot.id=follower2
