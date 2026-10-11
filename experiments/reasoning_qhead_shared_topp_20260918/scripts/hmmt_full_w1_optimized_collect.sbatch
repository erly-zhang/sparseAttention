#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=hmmt-full-w1-opt-collect
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
RUN_ROOT="${1:?}"
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/hmmt_full_w1_optimized.py" collect "$RUN_ROOT"
