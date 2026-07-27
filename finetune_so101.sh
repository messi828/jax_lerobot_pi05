#!/usr/bin/env bash
# SO101 (SO-ARM101) pi0.5 微调流程
# 数据集：jaylen/pick_lift_cube_so101_v2（本地 HF 缓存，由 record_with_leader.py 采集）
# 显存不足时改用 TASK="pi05_so101_lora"

set -e

TASK="pi05_so101"
EXP_NAME="so101_pick"

# 1) 计算归一化统计（写入 ./assets/$TASK/<repo_id>/）
python scripts/compute_norm_stats.py --config-name $TASK

# 2) 训练（checkpoint 输出 ./checkpoints/$TASK/$EXP_NAME/<step>/）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py $TASK \
    --exp-name $EXP_NAME \
    --overwrite

# 3) 训练完成后启动推理服务器：
# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=$TASK \
#     --policy.dir=checkpoints/$TASK/$EXP_NAME/<step>
