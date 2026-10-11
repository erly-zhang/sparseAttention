#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=low
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=jspace-free-collect
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
"$PY" -u "${1:?}/collect.py"
