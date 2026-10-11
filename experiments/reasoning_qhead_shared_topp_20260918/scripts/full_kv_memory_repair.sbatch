#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=kv-memory-repair
set -euo pipefail
module purge
module load wulver
R="${1:?root}"; PHASE="${2:?phase}"
export PYTHONPATH="$R/source/ra_harness" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data QHEAD_DEBUG=0
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE
exec /project/yl258/jz648/miniconda3/envs/random-attention/bin/python "$R/source/scripts/run_full_kv_memory_repair.py" "$R" "$PHASE" --worker "${SLURM_ARRAY_TASK_ID:-0}"
