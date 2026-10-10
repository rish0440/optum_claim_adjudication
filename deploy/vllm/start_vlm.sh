#!/usr/bin/env bash
# Manual Phase 6 launcher. Explicit mode is required; never fall back to base.
set -euo pipefail
if [[ $# -lt 1 || ( "$1" != "base" && "$1" != "adapter" ) ]]; then
  echo "Usage: bash deploy/vllm/start_vlm.sh {base|adapter} [server options]" >&2
  exit 2
fi
mode="$1"
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${VLM_PYTHON:-}" ]]; then
  vlm_python="$VLM_PYTHON"
else
  vlm_python="$(command -v python3 || command -v python || true)"
fi
if [[ ! -x "$vlm_python" ]]; then
  echo "ERROR: set VLM_PYTHON to the existing vLLM environment interpreter, or activate that environment first" >&2
  exit 1
fi
cd "$project_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$vlm_python" -u -m src.inference.vllm_server --mode "$mode" "$@"
