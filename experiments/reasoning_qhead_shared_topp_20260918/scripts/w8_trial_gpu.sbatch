#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=w8-math20-pair
set -euo pipefail
ROOT="${1:?}"
STAGE="${2:?}"
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/source/scripts/run_w8_trial.py" "$ROOT" "$STAGE"
