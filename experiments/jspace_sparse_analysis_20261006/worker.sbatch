#!/bin/bash
#SBATCH --job-name=jspace-decomp
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jspace_sparse_analysis_20261006/slurm-%A_%a.out
set -euo pipefail
ANALYSIS_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jspace_sparse_analysis_20261006
source "$ANALYSIS_RUN/../jlens_prefill_analysis_20261005/env.sh"
cd "$ANALYSIS_RUN"
exec "$BASE/env/bin/python" -u worker.py --index "$SLURM_ARRAY_TASK_ID"
