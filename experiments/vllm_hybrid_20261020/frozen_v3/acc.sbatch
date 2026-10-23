#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=v3-acc
# usage: --export=ALL,ARM=..,K=..,RUN=0|1,SEQS=..   (one engine per run)
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v3/common_env.sh
OUT=$STUDY/results_v2/acc_math/${ARM}_K${K}; mkdir -p $OUT/meta_run$RUN; verify_frozen $OUT/meta_run$RUN
hostname > $OUT/meta_run$RUN/host.txt
env $(arm_env $ARM $K) python $FROZEN/vllm_task_acc.py --mode generate --arm $ARM --task math \
  --ids $STUDY/inputs_ids/math.jsonl --runs 2 --run-index $RUN --max-num-seqs $SEQS --out $OUT > $OUT/log_run$RUN.txt 2>&1
echo rc=$?; check_log $OUT/log_run$RUN.txt $ARM | tee $OUT/check_run$RUN.txt; grep -E "GENERATED|Traceback|Error" $OUT/log_run$RUN.txt | tail -3 | cut -c1-300
