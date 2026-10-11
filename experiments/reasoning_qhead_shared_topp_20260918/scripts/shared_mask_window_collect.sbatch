#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=mask-window-grade
set -euo pipefail
RUN_ROOT="${1:?Comparison root required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/summarize_window_compare.py" "$RUN_ROOT"
