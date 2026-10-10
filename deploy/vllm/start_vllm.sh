#!/usr/bin/env bash
# Legacy filename delegates to the guarded Phase 6 launcher. Explicit mode required.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/start_vlm.sh" "$@"
