#!/bin/bash
#SBATCH --job-name=jlens-replay-sparse
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
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005/slurm-replay-sparse-%A_%a.out
set -euo pipefail
ANALYSIS_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005
source "$ANALYSIS_RUN/env.sh"
cd "$ANALYSIS_RUN"
METHODS=(coverage_token_vertical coverage_token_vertical_slash random_token_vertical random_token_vertical_slash flexprefill minference)
method_index=$((SLURM_ARRAY_TASK_ID / 10))
work_shard=$((SLURM_ARRAY_TASK_ID % 10))
exec "$BASE/env/bin/python" -u replay.py --method "${METHODS[$method_index]}" --shard "$work_shard"
