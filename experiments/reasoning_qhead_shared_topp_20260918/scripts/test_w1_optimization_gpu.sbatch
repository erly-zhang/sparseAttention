#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=w1-opt-check
set -euo pipefail
E=/mmfs1/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918
OUT="$E/results/w1_optimization_exploration_20260927"
export PYTHONPATH="$E/ra_harness" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
"$PY" -m unittest discover -s "$E/tests" -p test_representative_optimized_shared_mask.py -v > "$OUT/optimization_correctness.log" 2>&1
"$PY" -u "$E/scripts/benchmark_w1_optimization.py" "$OUT" > "$OUT/benchmark.out" 2>&1
