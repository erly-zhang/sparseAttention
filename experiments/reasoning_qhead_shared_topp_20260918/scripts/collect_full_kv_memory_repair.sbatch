#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=kv-repair-collect
set -euo pipefail
module purge
module load wulver
R="${1:?root}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$R/source/ra_harness"
exec /project/yl258/jz648/miniconda3/envs/random-attention/bin/python "$R/source/scripts/check_full_kv_memory_repair.py" "$R" --phase formal
