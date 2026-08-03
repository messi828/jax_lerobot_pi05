#!/usr/bin/env bash
# Install dependencies and run full pi0.5 fine-tuning for the realshit task.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASK="pi05_so101_realshit"
EXP_NAME="${EXP_NAME:-realshit_full_v1}"
MODE="${MODE:-new}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
DATASET_REL="dataset/jaylen/realshit_so101_v1"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/autodl-tmp/openpi-cache}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

echo "[1/6] Check GPU, disk, and dataset"
nvidia-smi
df -h "$ROOT" /root/autodl-tmp 2>/dev/null || true
test -f "$DATASET_REL/meta/info.json" || {
  echo "ERROR: missing $ROOT/$DATASET_REL" >&2
  exit 1
}

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$OPENPI_DATA_HOME"

echo "[2/6] Install system dependencies"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential pkg-config ffmpeg curl tmux \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libavfilter-dev libswscale-dev libswresample-dev >/dev/null

echo "[3/6] Install uv and Python dependencies"
if ! command -v uv >/dev/null 2>&1; then
  if ! timeout 180 bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'; then
    python -m pip install --index-url "$UV_INDEX_URL" uv
  fi
  export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync --no-group rlds --no-dev
uv pip install -e ./lerobot/lerobot-main

echo "[4/6] Validate CUDA and the realshit dataset"
uv run python - <<'PY'
from pathlib import Path

import jax
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("dataset/jaylen/realshit_so101_v1").resolve()
ds = LeRobotDataset("jaylen/realshit_so101_v1", root=root)
sample = ds[0]
print("JAX devices:", jax.devices())
print("episodes/frames/fps:", ds.num_episodes, len(ds), ds.fps)
print("task:", sample["task"])
print("state/action:", sample["observation.state"].shape, sample["action"].shape)
print("front/wrist:", sample["observation.images.front"].shape, sample["observation.images.wrist"].shape)
assert ds.num_episodes == 64
assert len(ds) == 45950
assert ds.fps == 30
assert sample["task"] == "insert the plug into the socket"
assert tuple(sample["observation.state"].shape) == (6,)
assert tuple(sample["action"].shape) == (6,)
assert any(device.platform == "gpu" for device in jax.devices())
PY

echo "[5/6] Configuration"
echo "task=$TASK exp=$EXP_NAME mode=$MODE batch=$BATCH_SIZE steps=$NUM_TRAIN_STEPS save=$SAVE_INTERVAL"

echo "[6/6] Compute norm stats if needed, then train"
TASK="$TASK" \
EXP_NAME="$EXP_NAME" \
MODE="$MODE" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_TRAIN_STEPS="$NUM_TRAIN_STEPS" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
uv run bash finetune_realshit.sh

echo "Training finished: $ROOT/checkpoints/$TASK/$EXP_NAME"
