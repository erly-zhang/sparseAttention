#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=v2-repro
# usage: --export=ALL,ARM=..,K=..,FIX=0|1
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v2/common_env.sh
OUT=$STUDY/results_v2/validation/${ARM}_K${K}_fix${FIX}_rerun; mkdir -p $OUT; verify_frozen $OUT
env $(arm_env $ARM $K) TRIATTN_ASSERT_REACHABLE=1 TRIATTN_ASSERT_RAISE=0 HYBRID_CHECK=1 HYBRID_OVERFLOW_POLICY=fallback \
  python $FROZEN/vllm_task_acc.py --mode generate --arm $ARM --task math --ids $STUDY/inputs_ids/math_first200.jsonl \
  --runs 1 --run-index 0 --max-tokens 8192 --max-num-seqs 32 --out $OUT > $OUT/log.txt 2>&1
echo rc=$?; check_log $OUT/log.txt $ARM; grep -c "hybrid_attention: \[Hybrid\] overflow" $OUT/log.txt
