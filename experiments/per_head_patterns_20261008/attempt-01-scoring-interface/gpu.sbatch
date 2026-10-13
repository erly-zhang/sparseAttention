#!/bin/bash
#SBATCH --job-name=own-vs-prefill
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --time-min=02:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/per_head_patterns_20261008/slurm-%A_%a.out
set -euo pipefail
OWN_VS_ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/per_head_patterns_20261008
source "$OWN_VS_ROOT/env.sh"
cd "$OWN_VS_ROOT"
if [ "$1" = proof ]; then
    exec "$BASE/env/bin/python" -u execute.py proof
else
    exec "$BASE/env/bin/python" -u execute.py "$1" --index "${SLURM_ARRAY_TASK_ID}"
fi
