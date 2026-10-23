#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=vllm-smoke2
set -uo pipefail
module purge; module load wulver
WORK=/scratch/yl258/jz648/triattn_vllm
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh; conda activate $WORK/venv
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_CONFIG_PATH=/mmfs1/project/yl258/jz648/Random-Attention/scripts/vllm_rp_bench/logcfg.json
OUT=/mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/smoke2; mkdir -p $OUT
COMMON="--model /scratch/yl258/jz648/Qwen3-4B --dtype bfloat16 --max-model-len 4096 --enforce-eager --no-enable-prefix-caching --input-len 256 --output-len 2048 --batch-size 4 --num-iters 1 --num-iters-warmup 1"
env ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=1024 \
  TRIATTN_RUNTIME_DIVIDE_LENGTH=128 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 \
  vllm bench latency $COMMON > $OUT/rp_latency.log 2>&1
env ENABLE_TRIATTENTION=0 vllm bench latency $COMMON > $OUT/off_latency.log 2>&1
for f in rp off; do echo "$f applied=$(grep -c 'compression applied' $OUT/${f}_latency.log) noop=$(grep -c no_compactable_groups $OUT/${f}_latency.log) activated=$(grep -c 'plugin activated' $OUT/${f}_latency.log) $(grep -E 'Avg latency' $OUT/${f}_latency.log)"; done
