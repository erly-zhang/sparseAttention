#!/usr/bin/env bash
# Single-Cluster shared sparse mask experiment (two-phase task eval).
#
# Phase 1: head selection (default 3 samples, distinct domains)
# Phase 2: task eval with fixed global representative heads (ff/sf/fs/ss)
#
# Not limited to 10 samples — override via env:
#   EVAL_N=50              # default partial eval
#   EVAL_N=500 USE_ALL_RECORDS=true   # full LongBench-v2 (503 = 3 head + 500 eval)
#   MAX_TOTAL_TOKENS=131072           # 128k truncation budget
#   SKIP_DATA_BUILD=true   # reuse existing DATA_OUT
#
# Usually invoked via:
#   experiments/run_both_eval_sequential.sh          (32k)
#   experiments/run_full_longbench_128k_pipeline.sh  (128k full)
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/attentionmap/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/work/model/Qwen2.5-7B}"
DATA_OUT="${DATA_OUT:-/home/ubuntu/work/experiments/data/longbench_v2_32k_eval53_7b.jsonl}"
EXP_OUT="${EXP_OUT:-/home/ubuntu/work/experiments/outputs/shared_layer_mask_task_eval50_7b}"

HEAD_N="${HEAD_N:-3}"
EVAL_N="${EVAL_N:-50}"
TOP_P="${TOP_P:-0.95}"
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS:-ff,sf,fs,ss}"
# Token budget: prompts shorter than this are kept intact; longer contexts are truncated.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-${MAX_TOTAL_TOKENS}}"
TOKEN_MARGIN="${TOKEN_MARGIN:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
# Set USE_ALL_RECORDS=true to include all LongBench-v2 records (503 total: 3 head + 500 eval).
USE_ALL_RECORDS="${USE_ALL_RECORDS:-false}"
SKIP_DATA_BUILD="${SKIP_DATA_BUILD:-false}"

# Head selection: 3 different domains (1 sample each).
HEAD_DOMAINS=(
  "Long In-context Learning"
  "Single-Document QA"
  "Code Repository Understanding"
)

# Task eval: round-robin across all 6 LongBench-v2 domains for max coverage.
EVAL_DOMAINS=(
  "Long In-context Learning"
  "Single-Document QA"
  "Code Repository Understanding"
  "Multi-Document QA"
  "Long Structured Data Understanding"
  "Long-dialogue History Understanding"
)

echo "==> Model: ${MODEL_PATH}"
echo "==> Data:  ${DATA_OUT}"
echo "==> Out:   ${EXP_OUT}"

if [[ "${SKIP_DATA_BUILD:-false}" != "true" ]]; then
echo "==> Step 1: build two-phase dataset (head=${HEAD_N} distinct domains + eval=${EVAL_N})"
BUILD_ARGS=(
  --out "${DATA_OUT}"
  --max_total_tokens "${MAX_TOTAL_TOKENS}"
  --margin "${TOKEN_MARGIN}"
  --model_path "${MODEL_PATH}"
  --head_selection_num_samples "${HEAD_N}"
  --head_selection_domains "${HEAD_DOMAINS[@]}"
  --eval_domains "${EVAL_DOMAINS[@]}"
  --seed 42
)
if [[ "${USE_ALL_RECORDS}" == "true" ]]; then
  BUILD_ARGS+=(--use_all_records)
else
  BUILD_ARGS+=(--eval_num_samples "${EVAL_N}")
fi
"${PYTHON}" experiments/build_longbench_v2_32k_domain_sample.py "${BUILD_ARGS[@]}"
else
  echo "==> Step 1: SKIP_DATA_BUILD=true, reusing ${DATA_OUT}"
fi

if [[ "${USE_ALL_RECORDS}" == "true" ]]; then
  EVAL_N="$("${PYTHON}" -c "import json; print(json.load(open('${DATA_OUT%.jsonl}.selection.json'))['eval_num_samples'])")"
  echo "==> USE_ALL_RECORDS: eval_num_samples=${EVAL_N}"
fi

echo "==> Step 2: phase-1 head selection (${HEAD_N}) + phase-2 sparse eval (${EVAL_N})"
"${PYTHON}" experiments/run_shared_layer_mask_experiment.py \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_OUT}" \
  --output_dir "${EXP_OUT}" \
  --num_samples "$((HEAD_N + EVAL_N))" \
  --head_selection_num_samples "${HEAD_N}" \
  --eval_num_samples "${EVAL_N}" \
  --max_input_length "${MAX_INPUT_LENGTH}" \
  --last_q 32 \
  --chunk_size "${CHUNK_SIZE}" \
  --mask_method top_p \
  --top_p "${TOP_P}" \
  --apply_prefill false \
  --apply_decode false \
  --run_task_eval true \
  --eval_apply_prefill true \
  --eval_apply_decode true \
  --eval_max_new_tokens 8 \
  --eval_mode_combos "${EVAL_MODE_COMBOS}" \
  --save_masks true \
  --save_similarity false \
  --filter_after_run true

echo "==> Done."
echo "Selection report: ${DATA_OUT%.jsonl}.selection.json"
echo "Global heads: ${EXP_OUT}/global_representative_heads.json"
echo "Task eval summary: ${EXP_OUT}/task_eval_summary.json"
