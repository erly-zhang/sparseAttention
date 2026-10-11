#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=w1-rec-smoke-gate
set -euo pipefail
CONTROL="${1:?Controller directory required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$CONTROL/scripts/submit_w1_after_smoke.py" "$CONTROL"
