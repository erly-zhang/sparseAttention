#!/bin/bash
# Add isolated analysis dependencies without changing the frozen runtime.
JLENS_ANALYSIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$JLENS_ANALYSIS_DIR/../representative_patterns_fast_retry03_20261001/env.sh"
export PYTHONPATH="$JLENS_ANALYSIS_DIR/deps:/ocean/projects/cis260309p/jzhang114/jacobian-lens:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_CACHE_DIR="$JLENS_ANALYSIS_DIR/triton-cache-${SLURM_JOB_ID:-cpu}"
