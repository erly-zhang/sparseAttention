#!/bin/bash
#SBATCH --job-name=masa-full128
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
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/masa_full_query_share_20261012/attempt02/slurm-%A_%a.out
set -euo pipefail
source /ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/masa_full_query_share_20261012/attempt02/env.sh
cd "$MASA_EQUAL_ROOT"
if [ "$1" = proof ]; then
    exec "$BASE/env/bin/python" -u worker.py proof
else
    exec "$BASE/env/bin/python" -u worker.py formal --index "$SLURM_ARRAY_TASK_ID"
fi
