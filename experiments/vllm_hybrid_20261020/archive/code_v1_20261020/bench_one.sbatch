#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=vllm-thr
# usage: sbatch --export=ALL,ARM=..,K=..,IL=..,OL=..,NP=..,SEQS=..,TAG=.. bench_one.sbatch
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/common_env.sh
STATS=/mmfs1/project/yl258/jz648/Random-Attention/kvcompress/harness/stats_pl/qwen3-4b.pt
OUT=$STUDY/results/$TAG; mkdir -p $OUT
NAME=${ARM}_K${K}_in${IL}_out${OL}_n${NP}_s${SEQS}
[ -e $OUT/$NAME.json ] && { echo "exists: $OUT/$NAME.json"; exit 0; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $OUT/$NAME.gpu
env $(arm_env $ARM $K $STATS) python $STUDY/bench_throughput.py --arm $ARM --input-len $IL --output-len $OL \
  --num-prompts $NP --max-num-seqs $SEQS --out $OUT/$NAME.json.tmp > $OUT/$NAME.log 2>&1
rc=$?
check_log $OUT/$NAME.log $ARM | tee $OUT/$NAME.check
grep -E "GPU KV cache size|Maximum concurrency" $OUT/$NAME.log | head -3
[ $rc = 0 ] && mv $OUT/$NAME.json.tmp $OUT/$NAME.json
echo rc=$rc; grep RESULT $OUT/$NAME.log
