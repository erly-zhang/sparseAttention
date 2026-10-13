#!/bin/bash
KV_CAUSAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$KV_CAUSAL_DIR/../jlens_prefill_analysis_20261005/env.sh"

export TRITON_CACHE_DIR="$KV_CAUSAL_DIR/triton-cache-${SLURM_JOB_ID:-cpu}"
