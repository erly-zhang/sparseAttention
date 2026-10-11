#!/bin/bash
#SBATCH --job-name=residual-score-check
#SBATCH --partition=gpu
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --exclude=n0046
#SBATCH --output=/mmfs1/project/yl258/jz648/sparseAttention/experiments/outputs/random_score_residual_20260910/cross-gpu-%j.log
set -euo pipefail
cd /mmfs1/project/yl258/jz648/sparseAttention
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="$PWD/.cache/triton-cross-$SLURM_JOB_ID"
nvidia-smi --query-gpu=uuid,name --format=csv
/project/yl258/jz648/miniconda3/envs/sparseattention-score26/bin/python -u \
    -m experiments.random_score_residual_20260910.test_correctness \
    --fingerprint-only \
    --output "$PWD/experiments/outputs/random_score_residual_20260910/cross_gpu_$SLURM_JOB_ID.json"
