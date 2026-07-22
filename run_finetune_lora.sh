#!/usr/bin/env bash
# LoRA fine-tune of pi05_base on the local libero_object dataset.
#
# Companion to run_finetune.sh (full FT). LoRA trains only low-rank adapters (base
# weights frozen via freeze_filter), so it needs >22.5GB rather than >70GB and fits
# the ~32GB free per GPU on this shared box -- no FSDP sharding required.
#
# Two stages: (1) compute normalization stats (CPU, safe while GPUs are busy),
#             (2) launch LoRA training on a single GPU.
#
# Usage:
#   ./run_finetune_lora.sh                  # GPU 7, defaults below
#   GPU=3 ./run_finetune_lora.sh            # pick a GPU
#   STEPS=5000 BATCH_SIZE=16 ./run_finetune_lora.sh
#   ./run_finetune_lora.sh --stats-only     # just compute norm stats, don't train
#
# Env overrides: EXP_NAME, GPU, BATCH_SIZE, STEPS

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/mxy/openpi/.venv/bin/python"
export PYTHONPATH="${HERE}/src"          # use THIS worktree's openpi (has the pi05_libero_object_lora config)

CONFIG="pi05_libero_object_lora"
EXP_NAME="${EXP_NAME:-libero_object_lora}"
GPU="${GPU:-7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"

cd "${HERE}"

# ---- Stage 1: normalization statistics (CPU only; harmless while GPUs are occupied) ----
STATS="${HERE}/assets/${CONFIG}/libero_object/norm_stats.json"   # ./assets/<config>/libero_object/norm_stats.json
if [[ -f "${STATS}" ]]; then
  echo "[1/2] norm stats already present: ${STATS}"
else
  echo "[1/2] computing norm stats (CPU)..."
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG}"
fi

if [[ "${1:-}" == "--stats-only" ]]; then
  echo "stats-only requested; not training."
  exit 0
fi

# ---- Stage 2: LoRA fine-tune on one GPU ----
echo "[2/2] launching LoRA FT | gpu=${GPU} batch=${BATCH_SIZE} steps=${STEPS} exp=${EXP_NAME}"
# XLA_PYTHON_CLIENT_PREALLOCATE=false: allocate on demand instead of grabbing a fixed
# fraction of the card up front. Essential on this SHARED box -- other jobs hold ~46GB,
# so the README's MEM_FRACTION=0.9 (0.9*80=72GB) would OOM immediately. On demand, JAX
# takes only what LoRA needs (~25GB), which fits the ~32GB free.
CUDA_VISIBLE_DEVICES="${GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PYTHON}" scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --batch-size="${BATCH_SIZE}" \
    --num-train-steps="${STEPS}" \
    --no-wandb-enabled \
    --overwrite

echo "done. checkpoints under: ${HERE}/checkpoints/${CONFIG}/${EXP_NAME}/"
