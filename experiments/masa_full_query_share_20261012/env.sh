#!/bin/bash
MASA_EQUAL_ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/masa_full_query_share_20261012
source "$MASA_EQUAL_ROOT/../psc_kl_quality_retry02_20260927/env.sh"
export PYTHONPATH="$MASA_EQUAL_ROOT/frozen:$MASA_EQUAL_ROOT:$PYTHONPATH"
export TRITON_CACHE_DIR="$MASA_EQUAL_ROOT/triton-cache-${SLURM_JOB_ID:-cpu}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
