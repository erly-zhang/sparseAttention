#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=rand1024-rec-grade
set -euo pipefail
RUN_ROOT="${1:?Frozen root required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/collect_random_fixed_recent_full.py" "$RUN_ROOT"
