#!/bin/bash
#SBATCH --job-name=jlens-smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:h100-80:1
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005/slurm-smoke-%A_%a.out
set -euo pipefail
ANALYSIS_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005
source "$ANALYSIS_RUN/env.sh"
cd "$ANALYSIS_RUN"
exec "$BASE/env/bin/python" -u validation.py smoke
