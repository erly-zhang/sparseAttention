#!/bin/bash
#SBATCH --job-name=rep-fast-audit
#SBATCH --partition=RM-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/representative_patterns_fast_retry01_20261001/slurm-audit-%A_%a.out
set -euo pipefail
RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/representative_patterns_fast_retry01_20261001
source "$RUN/env.sh"
cd "$RUN"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
exec "$BASE/env/bin/python" -u audit.py
