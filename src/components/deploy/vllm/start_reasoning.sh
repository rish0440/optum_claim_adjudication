#!/usr/bin/env bash
# Permanent equivalent of /tmp/optum-agent2-eval-server-40.sh.
# Same model, tokenizer, served names, decoding defaults and GPU allocation.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
reasoning_python="${REASONING_PYTHON:-${VLM_PYTHON:-$HOME/.venv-vllm/bin/python}}"
if [[ ! -x "$reasoning_python" ]]; then
  echo 'Set REASONING_PYTHON to the existing vLLM environment interpreter.' >&2
  exit 1
fi
cd "$project_root"
# Reuse the production CPU identity/rank/checksum resolver. No base fallback.
adapter="$("$reasoning_python" -c 'from src.config.reasoning import successful_reasoning_adapter; print(successful_reasoning_adapter()[0])')"
if [[ ! -f "$adapter/tokenizer_config.json" || ! -f "$adapter/tokenizer.json" ]]; then
  echo 'Successful reasoning adapter is missing its saved tokenizer.' >&2
  exit 1
fi
exec "$reasoning_python" -u -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --tokenizer "$adapter" \
  --served-model-name reasoning-base --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --max-model-len 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.40 --generation-config vllm \
  --enable-lora --max-lora-rank 8 --max-loras 1 \
  --lora-modules "reasoning-dpo=$adapter"
