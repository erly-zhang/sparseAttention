#!/bin/bash
PER_HEAD_PATTERN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PER_HEAD_PATTERN_DIR/../representative_patterns_fast_retry03_20261001/env.sh"
export PYTHONPATH="$PER_HEAD_PATTERN_DIR:$PYTHONPATH"
export TRITON_CACHE_DIR="$PER_HEAD_PATTERN_DIR/triton-cache-${SLURM_JOB_ID:-cpu}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
