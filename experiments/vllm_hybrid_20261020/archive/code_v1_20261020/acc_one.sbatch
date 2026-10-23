#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=vllm-acc
# usage: sbatch --export=ALL,ARM=..,K=..,TASK=..,SEQS=.. acc_one.sbatch
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/common_env.sh
STATS=/mmfs1/project/yl258/jz648/Random-Attention/kvcompress/harness/stats_pl/qwen3-4b.pt
OUT=$STUDY/results/acc_${TASK}/${ARM}_K${K}; mkdir -p $OUT
env $(arm_env $ARM $K $STATS) python $STUDY/vllm_task_acc.py --mode generate --arm $ARM --task $TASK \
  --ids $STUDY/inputs_ids/$TASK.jsonl --max-num-seqs $SEQS --out $OUT > $OUT/log.txt 2>&1
echo rc=$?; check_log $OUT/log.txt $ARM; grep -E "GENERATED|Traceback|Error|overflow" $OUT/log.txt | tail -5
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python $STUDY/vllm_task_acc.py --mode grade --task $TASK --out $OUT > $OUT/grade.txt 2>&1
tail -1 $OUT/grade.txt
