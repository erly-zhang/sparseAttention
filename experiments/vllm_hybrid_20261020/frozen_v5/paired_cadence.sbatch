#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=v5-cadence
# Same-GPU control: random_pp compacting every 64 vs every 1024 tokens, and hybrid, run back to back.
# usage: --export=ALL,K=..,IL=..,OL=..,NP=..,SEQS=..,TAG=..
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v5/common_env.sh
OUT=$STUDY/results_share/cadence_$TAG; mkdir -p $OUT/meta_$SLURM_JOB_ID; verify_frozen $OUT/meta_$SLURM_JOB_ID; hostname > $OUT/meta_$SLURM_JOB_ID/host.txt
for ARM in rp rp_d1024 hybrid; do
  NAME=${ARM}_K${K}_job$SLURM_JOB_ID
  env $(arm_env $ARM $K) python $FROZEN/bench_throughput.py --arm $ARM --input-len $IL --output-len $OL \
    --num-prompts $NP --max-num-seqs $SEQS --out $OUT/$NAME.json > $OUT/$NAME.log 2>&1
  echo "$NAME rc=$? $(check_log $OUT/$NAME.log $ARM | cut -c1-110) $(grep -o '"output_tok_per_s": [0-9.]*' $OUT/$NAME.json)"
done
