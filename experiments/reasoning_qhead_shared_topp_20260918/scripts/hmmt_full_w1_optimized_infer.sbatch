#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=hmmt-full-w1-opt
#SBATCH --exclude=n0068,n0090,n0049
#SBATCH --array=0-3%2
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN_ROOT="${1:?}"
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/hmmt_full_w1_optimized.py" infer "$RUN_ROOT" "${SLURM_ARRAY_TASK_ID:?}"
