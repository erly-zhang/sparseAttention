#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=compact-multibench
set -euo pipefail
ROOT="${1:?}"; PHASE="${2:?}"
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/source/scripts/run_compact_multibench.py" "$ROOT" "$PHASE"
