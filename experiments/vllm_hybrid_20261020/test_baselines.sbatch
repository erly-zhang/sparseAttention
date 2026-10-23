#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=bl-tests
module purge; module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
cd /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020
echo "== upstream equivalence (HF env, CPU+GPU)"; /project/yl258/jz648/miniconda3/envs/random-attention/bin/python src/tests/test_baseline_scoring.py 2>&1 | tail -3
conda activate /scratch/yl258/jz648/triattn_vllm/venv; export PYTHONNOUSERSITE=1
echo "== vLLM glue"; PYTHONPATH=/scratch/yl258/jz648/triattn_vllm/tri_baselines_fix python src/tests/test_baselines_vllm.py 2>&1 | tail -12
