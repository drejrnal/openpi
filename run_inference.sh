#!/usr/bin/env bash
# Run the pi05_base inference smoke test on a chosen GPU.
#
# Usage:
#   ./run_inference.sh          # default GPU (7)
#   ./run_inference.sh 3        # run on GPU 3
#   GPU=3 ./run_inference.sh    # same, via env var
#
# The checkpoint is already in ~/.cache/openpi, so this only loads the model
# and runs one forward pass -- a couple of minutes.

set -euo pipefail

# GPU: first positional arg, else $GPU, else 7.
GPU="${1:-${GPU:-7}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/mxy/openpi/.venv/bin/python"

echo "=== free memory on GPU ${GPU} before launch:"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader -i "${GPU}"
echo "=== launching inference on GPU ${GPU}..."

# CUDA_VISIBLE_DEVICES         -> expose only the chosen card (renumbered to 0 inside the process)
# XLA_PYTHON_CLIENT_PREALLOCATE -> stop JAX grabbing 75% up front; allocate on demand (~10 GB)
# PYTHONPATH                    -> use THIS worktree's src, not the venv's installed openpi
cd "${HERE}"
CUDA_VISIBLE_DEVICES="${GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="${HERE}/src" \
  "${PYTHON}" -u pi_inference_test.py 2>&1 | tee infer.log
