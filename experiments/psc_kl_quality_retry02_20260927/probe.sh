#!/bin/bash
set -euo pipefail
HERE=/jet/home/jzhang114/vscode_cli/runtimefixed
exec > "$HERE/probe.log" 2>&1
source "$HERE/env.sh"
cd "$ROOT"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="$HERE/triton-cache"
nvidia-smi
timeout --signal=TERM --kill-after=10s 300s "$BASE/env/bin/python" -u "$HERE/probe.py"
