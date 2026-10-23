#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pool-abl-test
module purge; module load wulver
E=/mmfs1/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918
export PYTHONPATH=$E/ra_harness PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
cd $E/tests
$PY test_w1_hybrid_pool_ablation.py 2>&1 | tail -30
echo "== v6 regression"; $PY test_w1_hybrid_v6.py 2>&1 | tail -6
