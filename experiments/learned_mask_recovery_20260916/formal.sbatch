#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=/project/yl258/jz648/sparseAttention/experiments/outputs/learned_mask_recovery_%j.log
set -euo pipefail
cd /project/yl258/jz648/sparseAttention
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:$PWD/FlexPrefill"
exec /project/yl258/jz648/miniconda3/envs/sparseattention-score26/bin/python -u -m experiments.learned_mask_recovery_20260916.pipeline formal --variant "$1" --shard "$2"
