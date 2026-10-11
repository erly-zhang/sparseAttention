#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=hmmt-rpp-check
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
ROOT="${1:?}"
export PYTHONPATH="$ROOT/source/ra_harness"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/source/scripts/hmmt_official_random_pp.py" check "$ROOT"
