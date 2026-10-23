#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=hyb-kern
module purge; module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh; conda activate /scratch/yl258/jz648/triattn_vllm/venv
export PYTHONNOUSERSITE=1
cd /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020
python src/tests/test_hybrid_kernels.py
