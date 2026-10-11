#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=low
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=n0005,n0024,n0026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --job-name=jspace-free-generation
set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT="${1:?}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
"$PY" -u "$ROOT/free_generate.py" --sample "$SLURM_ARRAY_TASK_ID"
