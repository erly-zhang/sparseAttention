#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=low
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=n1547
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=jspace-gamma
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT="${1:?}"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/gamma_sensitivity.py"
