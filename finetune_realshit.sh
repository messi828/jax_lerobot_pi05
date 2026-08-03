#!/usr/bin/env bash
# pi0.5 fine-tuning for the realshit insertion task.
# Set TASK=pi05_so101_realshit_lora on a <=24GB GPU.
set -euo pipefail

TASK="${TASK:-pi05_so101_realshit_lora}"
EXP_NAME="${EXP_NAME:-realshit}"
MODE="${MODE:-new}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"

case "$TASK" in
  pi05_so101_realshit|pi05_so101_realshit_lora) ;;
  *) echo "TASK must be pi05_so101_realshit or pi05_so101_realshit_lora" >&2; exit 2 ;;
esac

case "$MODE" in
  new) RUN_MODE=() ;;
  resume) RUN_MODE=(--resume) ;;
  overwrite) RUN_MODE=(--overwrite) ;;
  *) echo "MODE must be new, resume, or overwrite" >&2; exit 2 ;;
esac

if [[ ! -f "assets/$TASK/jaylen/realshit_so101_v1/norm_stats.json" ]]; then
  python scripts/compute_norm_stats.py --config-name "$TASK"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
  python scripts/train.py "$TASK" \
    --exp-name "$EXP_NAME" \
    --batch-size "$BATCH_SIZE" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    "${RUN_MODE[@]}"
