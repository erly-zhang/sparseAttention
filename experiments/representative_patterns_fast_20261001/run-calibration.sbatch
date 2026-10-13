#!/bin/bash
#SBATCH --job-name=rep-fast-calibration
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:h100-80:1
#SBATCH --array=0-2%3
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/representative_patterns_fast_20261001/slurm-calibration-%A_%a.out
set -euo pipefail
RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/representative_patterns_fast_20261001
source "$RUN/env.sh"
cd "$RUN"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_CACHE_DIR="$RUN/triton-cache-${SLURM_JOB_ID}"
exec "$BASE/env/bin/python" -u run.py run --profile calibration
