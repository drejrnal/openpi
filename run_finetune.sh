#!/usr/bin/env bash
# Full fine-tune of pi05_droid_jointpos on the local Franka LeRobot v3
# dataset, with FSDP and Weights & Biases logging.
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
# Before the first training run:
#   uv run wandb login
#
# Env overrides: EXP_NAME, GPUS, FSDP_DEVICES, BATCH_SIZE, STEPS,
# PROJECT_NAME, MEM_FRACTION, PYTHON.
#
# A full pi05 fine-tune needs >70GB on one card, so it must be sharded. Set
# FSDP_DEVICES to the number of visible GPUs. The default 0.35 memory fraction
# caps JAX at about 28GB on each 80GB A100, leaving room for existing processes.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${HERE}/.venv/bin/python}"
export PYTHONPATH="${HERE}/src"

CONFIG="pi05_franka_finetune"
EXP_NAME="${EXP_NAME:-franka_object_full}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"
PROJECT_NAME="${PROJECT_NAME:-openpi-franka}"
MEM_FRACTION="${MEM_FRACTION:-0.35}"

cd "${HERE}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found at ${PYTHON}." >&2
  echo "Run 'uv sync --group dev' in ${HERE}, then retry." >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne "${FSDP_DEVICES}" ]]; then
  echo "FSDP_DEVICES=${FSDP_DEVICES} must equal the number of GPUs in GPUS=${GPUS}." >&2
  exit 1
fi
if (( BATCH_SIZE % ${#GPU_LIST[@]} != 0 )); then
  echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by ${#GPU_LIST[@]} visible GPUs." >&2
  exit 1
fi

# ---- Stage 1: normalization statistics (CPU only; harmless while GPUs are occupied) ----
STATS="${HERE}/assets/${CONFIG}/franka_object/norm_stats.json"
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
echo "[2/2] launching Franka full FT"
echo "      gpus=${GPUS} fsdp=${FSDP_DEVICES} batch=${BATCH_SIZE} steps=${STEPS}"
echo "      exp=${EXP_NAME} wandb_project=${PROJECT_NAME} mem_fraction=${MEM_FRACTION}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}" \
  "${PYTHON}" scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --fsdp-devices="${FSDP_DEVICES}" \
    --batch-size="${BATCH_SIZE}" \
    --num-train-steps="${STEPS}" \
    --project-name="${PROJECT_NAME}" \
    --wandb-enabled \
    --overwrite

echo "done. checkpoints under: ${HERE}/checkpoints/${CONFIG}/${EXP_NAME}/"
