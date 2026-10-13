#!/bin/bash
#SBATCH --job-name=jlens-kv-causal
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
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_kv_interventions_20261007/slurm-%A_%a.out
set -euo pipefail
KV_CAUSAL_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_kv_interventions_20261007
source "$KV_CAUSAL_RUN/env.sh"
cd "$KV_CAUSAL_RUN"
"$BASE/env/bin/python" -u run.py --index "${SLURM_ARRAY_TASK_ID}"
exec "$BASE/env/bin/python" -u analyze.py --index "${SLURM_ARRAY_TASK_ID}"
