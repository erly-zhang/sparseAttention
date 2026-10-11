#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/project/yl258/jz648/sparseAttention/experiments/outputs/learned_mask_nnonly_%j.log
set -euo pipefail
cd /project/yl258/jz648/sparseAttention
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/FlexPrefill"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
/project/yl258/jz648/miniconda3/envs/sparseattention-score26/bin/python -u -m experiments.learned_mask_nnonly_20260914.pipeline "$1"
