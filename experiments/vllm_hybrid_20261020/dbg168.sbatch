#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=dbg168
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v3/common_env.sh
OUT=$STUDY/results_v2/debug_idx168; mkdir -p $OUT
env $(arm_env snapkv 1024) PYTHONPATH=/scratch/yl258/jz648/triattn_vllm/tri_baselines_dbg python $FROZEN/vllm_task_acc.py --mode generate --arm snapkv --task math \
  --ids $STUDY/inputs_ids/math_idx168.jsonl --runs 1 --run-index 0 --max-tokens 4096 --max-num-seqs 8 --out $OUT > $OUT/log.txt 2>&1
echo rc=$?; grep -E "BASELINE-NOQUERIES|signal triggered|compression applied|compression skipped" $OUT/log.txt | head -12 | cut -c1-330
