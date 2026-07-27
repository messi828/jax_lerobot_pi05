#!/usr/bin/env bash
# 从臂（SO101 follower）重新标定
# 只连接从臂，不要接主臂；按提示把每个关节转到零点后确认

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower_black \
  --robot.id=black
