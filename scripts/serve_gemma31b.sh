#!/usr/bin/env bash
# Serve the copied 31B weights with the corrected QAT INT4 metadata.
set -euo pipefail

GEMMA_RUNTIME="${GEMMA_RUNTIME:-$HOME/.local/share/gemma31b}"
GEMMA_MODEL_DIR="${GEMMA_MODEL_DIR:-$GEMMA_RUNTIME/model}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "$GEMMA_RUNTIME/venv/bin/vllm" serve "$GEMMA_MODEL_DIR" \
  --served-model-name gemma4-31b \
  --host "${GEMMA_HOST:-0.0.0.0}" --port "${GEMMA_PORT:-8000}" \
  --max-model-len 8192 --max-num-seqs 2 \
  --gpu-memory-utilization "${GEMMA_GPU_MEMORY_UTILIZATION:-0.35}" \
  --attention-backend TRITON_ATTN --enforce-eager \
  --limit-mm-per-prompt '{"image":2,"audio":0,"video":0}' \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  "$@"
