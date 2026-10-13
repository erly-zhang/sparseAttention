#!/bin/bash
#SBATCH --job-name=jlens-speed
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:h100-80:1
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_fit_speed_20261006/slurm-%A_%a.out
set -euo pipefail
SPEED_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_fit_speed_20261006
source "$SPEED_RUN/../jlens_prefill_analysis_20261005/env.sh"
export TRITON_CACHE_DIR="$SPEED_RUN/triton-cache-${SLURM_JOB_ID}"
cd "$SPEED_RUN"
exec "$BASE/env/bin/python" -u benchmark.py --worker "$SLURM_ARRAY_TASK_ID"
