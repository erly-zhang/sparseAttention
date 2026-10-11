#!/bin/bash
set -euo pipefail
RUN_ROOT="${1:?Frozen root required}"
SHARD="${2:?Shard required}"
[[ "$SHARD" =~ ^[0-7]$ ]] || exit 2
PYTHON=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data PYTHONPATH="$RUN_ROOT/source/ra_harness" QHEAD_DEBUG=0
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE LOG_UNION EVICT_TIMING
OUT="$RUN_ROOT/formal/shard_${SHARD}"
mkdir "$OUT"
trap 'rc=$?; printf "{\"exit_code\":%s,\"ended_epoch\":%s}\n" "$rc" "$(date +%s)" > "$OUT/exit.json"' EXIT
"$PYTHON" "$RUN_ROOT/source/scripts/official_random_pp_preflight.py" "$RUN_ROOT" "$OUT" "$SHARD"
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used --format=csv > "$OUT/gpu_before.csv"
METHOD=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["method"])' "$RUN_ROOT/config.json")
"$PYTHON" -u "$RUN_ROOT/source/ra_harness/kvcompress/harness/eval_hf.py" \
 --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
 --data_name math --split test --prompt_type qwen-instruct --surround_with_messages \
 --attention_implementation flash_attention_2 --sparsity_method eviction \
 --eviction_mode "$METHOD" --token_budget 1024 --residual_length 64 \
 --mask_seed 42 --seed 42 \
 --batch_size 2 --run_id 0 --max_tokens 32768 --warmup_new_tokens 2 \
 --num_shards 8 --shard_id "$SHARD" --use_batch_exist --output_dir "$OUT" > "$OUT/runner.log" 2>&1
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used --format=csv > "$OUT/gpu_after.csv"
