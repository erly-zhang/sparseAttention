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
#SBATCH --time=00:30:00
#SBATCH --job-name=jspace-english-display
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT="${1:?}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
"$PY" -u "$ROOT/export_figure5.py" --display english --output "$ROOT/figure5_html_en_v2"
"$PY" -u "$ROOT/pack_figure5.py" --directory "$ROOT/figure5_html_en_v2"
"$PY" -u "$ROOT/validate_figure5_data.py" --directory "$ROOT/figure5_html_en_v2"
