#!/bin/bash
#SBATCH --job-name=flex-usage-diag
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/flexprefill_head_usage_retry01_20261002/diagnose-%j.out
set -euo pipefail
USAGE_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/flexprefill_head_usage_retry01_20261002
source "$USAGE_RUN/../representative_patterns_fast_retry03_20261001/env.sh"
cd "$USAGE_RUN"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_CACHE_DIR="$USAGE_RUN/diagnostic-triton-cache-${SLURM_JOB_ID}"
exec "$BASE/env/bin/python" -u diagnose.py
