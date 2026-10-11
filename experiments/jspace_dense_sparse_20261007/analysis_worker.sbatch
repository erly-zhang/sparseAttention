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
#SBATCH --time=02:00:00
#SBATCH --job-name=jspace-analysis
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT="${1:?}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
"$PY" -u "$ROOT/analyze.py"
"$PY" -u "$ROOT/make_figures.py"
"$PY" -u "$ROOT/build_report.py"
