#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=hyb-smoke
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/common_env.sh
OUT=$STUDY/smoke_hybrid_$SLURM_JOB_ID; mkdir -p $OUT
COMMON="--model /scratch/yl258/jz648/Qwen3-4B --dtype bfloat16 --max-model-len 8192 --enforce-eager --no-enable-prefix-caching --input-len 512 --output-len 6000 --batch-size 4 --num-iters 1 --num-iters-warmup 0"
env $(arm_env hybrid 1024) HYBRID_CHECK=1 vllm bench latency $COMMON > $OUT/hybrid_latency.log 2>&1
check_log $OUT/hybrid_latency.log hybrid
grep -E "\[Hybrid\]|HYBRID-SELECTOR|GPU KV cache size|Maximum concurrency" $OUT/hybrid_latency.log | head -20
grep -E "Traceback|Error" $OUT/hybrid_latency.log | head
