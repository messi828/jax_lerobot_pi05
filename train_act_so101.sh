#!/usr/bin/env bash
# SO101 ACT 训练（PyTorch lerobot，与 pi0.5 共用工作区 dataset/ 下同一份数据）
# 用法：conda activate jax_pi05 && bash train_act_so101.sh
# 换任务/数据集时改 REPO_ID 与 DATASET_ROOT

set -e

WORKSPACE="/home/jaylen/桌面/jax_lerobot_pi05"
REPO_ID="jaylen/pick_lift_cube_so101_v2"
DATASET_ROOT="$WORKSPACE/dataset/jaylen/pick_lift_cube_so101_v2"
JOB_NAME="act_so101_pick"
OUTPUT_DIR="$WORKSPACE/checkpoints/act/$JOB_NAME"

lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=2000 \
  --log_freq=100 \
  --wandb.enable=false

# 训练产物：$OUTPUT_DIR/checkpoints/<step>/pretrained_model/
#
# 断点续训：
#   lerobot-train \
#     --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" \
#     --resume=true

conda activate jax_pi05
cd /home/jaylen/桌面/jax_lerobot_pi05

lerobot-train \
  --config_path=checkpoints/act/act_so101_pick/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
