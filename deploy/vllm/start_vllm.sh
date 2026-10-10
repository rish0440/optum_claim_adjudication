#!/usr/bin/env bash
# Phase 5: run ONLY on a remote Linux NVIDIA GPU machine with vLLM installed.
# The trained adapter (not its 4-bit Colab training runtime) is loaded at startup.
# The server binds to localhost, so you must use an SSH tunnel to reach it.
set -euo pipefail

: "${VLLM_API_KEY:?Set VLLM_API_KEY on GPU server (never commit it)}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/models/qwen25-vl-invoice-lora/adapter}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
GPU_PORT="${GPU_PORT:-8001}"

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "ERROR: missing $ADAPTER_DIR/adapter_config.json; copy your trained adapter first" >&2
  exit 1
fi

# vLLM version/GPU combinations may need adjustments. Use a GPU with >=24GB
# VRAM if T4 runs out of memory. Only the text decoder LoRA was trained here.
exec vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$GPU_PORT" \
  --served-model-name qwen-base \
  --enable-lora \
  --lora-modules "invoice-extractor=$ADAPTER_DIR" \
  --max-lora-rank 8 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image":1}' \
  --enforce-eager
