#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --job-name=rand-compact1024
set -euo pipefail
RUN_ROOT="${1:?Frozen root required}"
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
export QHEAD_DEBUG=0
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/run_random_compact_recent_smoke.py" "$RUN_ROOT"
