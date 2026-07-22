#!/usr/bin/env bash
# Serve the fine-tuned pi05_libero_object checkpoint as a WebSocket policy server.
#
# Loads the trained checkpoint and listens on 0.0.0.0:<PORT> for a remote client
# (e.g. examples/libero/main.py driving the LIBERO simulator) to send observations
# and receive action chunks. The server binds all interfaces, so a client on another
# machine connects to THIS host's IP at PORT (make sure that port is open / reachable).
#
# Request  (client -> server): {observation/image, observation/wrist_image,
#                               observation/state(8), prompt}   (msgpack over websocket)
# Response (server -> client): {actions: (10, 7), server_timing, policy_timing}
#
# Usage:
#   ./run_server.sh                 # GPU 7, port 8000, final checkpoint (step 9999)
#   GPU=3 PORT=8080 ./run_server.sh
#   STEP=5000 ./run_server.sh       # serve the earlier checkpoint instead
#
# Env overrides: GPU, PORT, EXP_NAME, STEP

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/mxy/openpi/.venv/bin/python"
export PYTHONPATH="${HERE}/src"          # use THIS worktree's openpi (has the pi05_libero_object config)

CONFIG="pi05_libero_object"
EXP_NAME="${EXP_NAME:-libero_object_full_ft}"
STEP="${STEP:-9999}"
GPU="${GPU:-7}"
PORT="${PORT:-8000}"

CKPT="${HERE}/checkpoints/${CONFIG}/${EXP_NAME}/${STEP}"
if [[ ! -d "${CKPT}/params" ]]; then
  echo "ERROR: no checkpoint params at ${CKPT}" >&2
  echo "Available steps:" >&2
  ls "${HERE}/checkpoints/${CONFIG}/${EXP_NAME}" 2>/dev/null >&2 || true
  exit 1
fi

cd "${HERE}"
echo "serving ${CONFIG} @ ${CKPT}"
echo "listening on 0.0.0.0:${PORT} (this host's IP, port ${PORT}) -- clients connect there"

# XLA_PYTHON_CLIENT_PREALLOCATE=false: on-demand GPU allocation so it coexists with other
# jobs (inference needs only ~8-10GB). The server runs until you stop it (Ctrl-C / tmux kill).
CUDA_VISIBLE_DEVICES="${GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PYTHON}" scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${CONFIG}" \
    --policy.dir="${CKPT}"
