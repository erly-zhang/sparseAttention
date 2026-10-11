#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=w1-passcode-grade
set -euo pipefail
ROOT="${1:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$ROOT/source/scripts/collect_w1_passcode_probe.py" "$ROOT"
