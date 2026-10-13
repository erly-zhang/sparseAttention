#!/bin/bash
#SBATCH --job-name=baseline-sparsity
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:h100-80:1
#SBATCH --array=0-79%80
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/baseline_sparsity_20261007/slurm-worker-%A_%a.out
set -euo pipefail
RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/baseline_sparsity_20261007
source "$RUN/../jlens_prefill_analysis_20261005/env.sh"
export TRITON_CACHE_DIR="$RUN/triton-cache-${SLURM_JOB_ID}"
cd "$RUN"
exec "$BASE/env/bin/python" -u replay.py worker
