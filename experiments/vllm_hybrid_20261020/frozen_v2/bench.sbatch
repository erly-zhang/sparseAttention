#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=v2-thr
# usage: --export=ALL,ARM=..,K=..,IL=..,OL=..,NP=..,SEQS=..,TAG=..
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v2/common_env.sh
OUT=$STUDY/results_v2/$TAG; mkdir -p $OUT
NAME=${ARM}_K${K}_in${IL}_out${OL}_n${NP}_s${SEQS}
[ -e $OUT/$NAME.json ] && { echo "exists: $OUT/$NAME.json"; exit 0; }
mkdir -p $OUT/$NAME.meta; verify_frozen $OUT/$NAME.meta
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $OUT/$NAME.meta/gpu.txt; hostname >> $OUT/$NAME.meta/gpu.txt
env $(arm_env $ARM $K) python $FROZEN/bench_throughput.py --arm $ARM --input-len $IL --output-len $OL \
  --num-prompts $NP --max-num-seqs $SEQS --out $OUT/$NAME.json.tmp > $OUT/$NAME.log 2>&1
rc=$?
check_log $OUT/$NAME.log $ARM | tee $OUT/$NAME.check
[ $rc = 0 ] && mv $OUT/$NAME.json.tmp $OUT/$NAME.json
echo rc=$rc; grep RESULT $OUT/$NAME.log | cut -c1-300
