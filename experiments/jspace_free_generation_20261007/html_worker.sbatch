#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=low
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=n0026
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=jspace-own-html-export
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT="${1:?}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
PILOT=/mmfs1/project/yl258/jz648/sparseAttention/experiments/jspace_dense_sparse_20261007
"$PY" -u "$ROOT/export_own_figure5.py"
"$PY" -u "$PILOT/pack_figure5.py" --directory "$ROOT/figure5_html_en"
"$PY" -u "$PILOT/validate_figure5_data.py" --directory "$ROOT/figure5_html_en"
