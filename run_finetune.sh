#!/usr/bin/env bash
# Full fine-tune of pi05_base on the local libero_object dataset, with FSDP.
#
# Two stages: (1) compute normalization stats (CPU, safe while GPUs are busy),
# (2) launch FSDP training across the chosen GPUs.
#
# Usage:
#   ./run_finetune.sh                       # 8-GPU FSDP, defaults below
#   GPUS=0,1,2,3 FSDP_DEVICES=4 ./run_finetune.sh
#   STEPS=2000 BATCH_SIZE=16 ./run_finetune.sh
#   ./run_finetune.sh --stats-only          # just compute norm stats, don't train
#
# Env overrides: EXP_NAME, GPUS, FSDP_DEVICES, BATCH_SIZE, STEPS
#
# A full pi05 fine-tune needs >70GB on a single card, so it MUST be sharded: set
# FSDP_DEVICES to the number of GPUs in GPUS. With ~32GB free per GPU here, FSDP=8 puts
# a ~9GB shard on each card, which fits. Uses on-demand GPU allocation (not the README's
# 0.9 mem-fraction) so it coexists with the other jobs on this shared box.
# For a lighter alternative that fits on ONE gpu, use run_finetune_lora.sh instead.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/mxy/openpi/.venv/bin/python"
export PYTHONPATH="${HERE}/src"          # use THIS worktree's openpi (has the pi05_libero_object config)

CONFIG="pi05_libero_object"
EXP_NAME="${EXP_NAME:-libero_object_full_ft}"
GPUS="${GPUS:-0,1,2,3,4,5}"
FSDP_DEVICES="${FSDP_DEVICES:-6}"
BATCH_SIZE="${BATCH_SIZE:-192}"
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

# ---- Stage 2: FSDP full fine-tune ----
echo "[2/2] launching full FT | gpus=${GPUS} fsdp=${FSDP_DEVICES} batch=${BATCH_SIZE} steps=${STEPS} exp=${EXP_NAME}"
# NOTE: do NOT use the README's XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 here. That grabs
# 0.9*80=72GB per card up front, which OOMs immediately on this SHARED box (~46GB is
# already held by other jobs, only ~32GB free). Instead allocate on demand so JAX takes
# only each GPU's FSDP shard (~9GB with fsdp=8) plus activations, fitting the free memory.
# FSDP shards params/optimizer/gradients across FSDP_DEVICES, so aggregate GPU memory is
# what counts; ensure FSDP_DEVICES divides the GPU count and BATCH_SIZE divides the GPU count.
CUDA_VISIBLE_DEVICES="${GPUS}" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  "${PYTHON}" scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --fsdp-devices="${FSDP_DEVICES}" \
    --batch-size="${BATCH_SIZE}" \
    --num-train-steps="${STEPS}" \
    --no-wandb-enabled \
    --overwrite

echo "done. checkpoints under: ${HERE}/checkpoints/${CONFIG}/${EXP_NAME}/"
