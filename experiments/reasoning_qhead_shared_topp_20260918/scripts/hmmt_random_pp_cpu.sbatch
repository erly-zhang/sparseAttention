#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=hmmt-rpp-collect
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
ROOT="${1:?}"
ACTION="${2:?}"
export PYTHONPATH="$ROOT/source/ra_harness"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/source/scripts/hmmt_official_random_pp.py" "$ACTION" "$ROOT"
