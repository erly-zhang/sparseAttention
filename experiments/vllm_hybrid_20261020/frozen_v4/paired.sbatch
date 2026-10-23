#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=v4-paired
# Same-node paired throughput: shared and per-head hybrid alternated (ABAB) on one GPU.
# usage: --export=ALL,K=..,IL=..,OL=..,NP=..,SEQS=..,TAG=..
set -uo pipefail
source /mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/frozen_v4/common_env.sh
OUT=$STUDY/results_share/paired_$TAG; mkdir -p $OUT/meta_$SLURM_JOB_ID; verify_frozen $OUT/meta_$SLURM_JOB_ID; hostname > $OUT/meta_$SLURM_JOB_ID/host.txt
for rep in 1 2; do for ARM in hybrid hybrid_ph; do
  NAME=${ARM}_K${K}_rep${rep}_job$SLURM_JOB_ID
  env $(arm_env $ARM $K) python $FROZEN/bench_throughput.py --arm $ARM --input-len $IL --output-len $OL \
    --num-prompts $NP --max-num-seqs $SEQS --out $OUT/$NAME.json > $OUT/$NAME.log 2>&1
  echo "$NAME rc=$? $(check_log $OUT/$NAME.log $ARM | cut -c1-120) $(grep -o '"output_tok_per_s": [0-9.]*' $OUT/$NAME.json)"
done; done
