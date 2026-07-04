#!/usr/bin/env bash
# Full LongBench-v2 pipeline (defaults to 32k for quick testing):
# build data -> Single-Cluster -> Graph2Vec.
#
# Truncation: prompt <= MAX_TOTAL_TOKENS kept intact; longer contexts truncated.
#
# Usage:
#   nohup bash experiments/run_full_longbench_128k_pipeline.sh \
#     > experiments/outputs/run_full_longbench_128k_pipeline.log 2>&1 &
#
# Optional env:
#   SKIP_DATA_BUILD=true | SKIP_SINGLE_CLUSTER=true | SKIP_GRAPH2VEC=true
#   MODEL_PATH, MAX_TOTAL_TOKENS, CHUNK_SIZE, EVAL_MODE_COMBOS
#
# For 32k eval use instead:
#   bash experiments/run_both_eval_sequential.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/attentionmap/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/work/model/Qwen2.5-7B}"
# Default to 32k for faster smoke tests; override via env for 128k runs.
# Example:
#   MAX_TOTAL_TOKENS=131072 \
#   DATA_OUT=/home/ubuntu/work/experiments/data/longbench_v2_128k_full_7b.jsonl \
#   SINGLE_OUT=/home/ubuntu/work/experiments/outputs/shared_layer_mask_task_eval_full128k_7b \
#   GRAPH2VEC_OUT=/home/ubuntu/work/experiments/outputs/graph2vec_cluster2_task_eval_full128k_7b \
#   PIPELINE_LOG=/home/ubuntu/work/experiments/outputs/run_full_longbench_128k_pipeline.log \
#   bash experiments/run_full_longbench_128k_pipeline.sh
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-${MAX_TOTAL_TOKENS}}"
TOKEN_MARGIN="${TOKEN_MARGIN:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"

DATA_OUT="${DATA_OUT:-/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl}"
SINGLE_OUT="${SINGLE_OUT:-/home/ubuntu/work/experiments/outputs/shared_layer_mask_task_eval_full32k_7b}"
GRAPH2VEC_OUT="${GRAPH2VEC_OUT:-/home/ubuntu/work/experiments/outputs/graph2vec_cluster2_task_eval_full32k_7b}"
PIPELINE_LOG="${PIPELINE_LOG:-/home/ubuntu/work/experiments/outputs/run_full_longbench_32k_pipeline.log}"

HEAD_N="${HEAD_N:-3}"
USE_ALL_RECORDS="${USE_ALL_RECORDS:-true}"
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS:-ff,sf,fs,ss}"

mkdir -p "$(dirname "${DATA_OUT}")" "${SINGLE_OUT}" "${GRAPH2VEC_OUT}"

echo "============================================================"
echo "LongBench-v2 FULL eval pipeline @ ${MAX_TOTAL_TOKENS} tokens"
echo "Model:  ${MODEL_PATH}"
echo "Data:   ${DATA_OUT}"
echo "Single: ${SINGLE_OUT}"
echo "G2V:    ${GRAPH2VEC_OUT}"
echo "Chunk:  ${CHUNK_SIZE}"
echo "============================================================"

if [[ "${SKIP_DATA_BUILD:-false}" != "true" ]]; then
  echo "==> [0/2] Build full LongBench-v2 jsonl (all records, ${MAX_TOTAL_TOKENS} token budget)"
  BUILD_ARGS=(
    --out "${DATA_OUT}"
    --max_total_tokens "${MAX_TOTAL_TOKENS}"
    --margin "${TOKEN_MARGIN}"
    --model_path "${MODEL_PATH}"
    --head_selection_num_samples "${HEAD_N}"
    --head_selection_domains
    "Long In-context Learning"
    "Single-Document QA"
    "Code Repository Understanding"
    --use_all_records
    --seed 42
  )
  "${PYTHON}" experiments/build_longbench_v2_32k_domain_sample.py "${BUILD_ARGS[@]}"
else
  echo "==> [0/2] SKIP_DATA_BUILD=true, reusing ${DATA_OUT}"
fi

EVAL_N="$("${PYTHON}" -c "import json; print(json.load(open('${DATA_OUT%.jsonl}.selection.json'))['eval_num_samples'])")"
TOTAL_N="$((HEAD_N + EVAL_N))"
echo "==> Dataset ready: head=${HEAD_N}, eval=${EVAL_N}, total=${TOTAL_N}"

if [[ "${SKIP_SINGLE_CLUSTER:-false}" != "true" ]]; then
  echo "==> [1/2] Single-Cluster shared mask eval"
  DATA_OUT="${DATA_OUT}" \
  EXP_OUT="${SINGLE_OUT}" \
  HEAD_N="${HEAD_N}" \
  EVAL_N="${EVAL_N}" \
  USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
  SKIP_DATA_BUILD=true \
  bash experiments/run_task_eval_10samples.sh 2>&1 | tee "${SINGLE_OUT}/run.log"
else
  echo "==> [1/2] SKIP_SINGLE_CLUSTER=true"
fi

if [[ "${SKIP_GRAPH2VEC:-false}" != "true" ]]; then
  echo "==> [2/2] Graph2Vec 2-Cluster shared mask eval"
  DATA_OUT="${DATA_OUT}" \
  EXP_OUT="${GRAPH2VEC_OUT}" \
  LOG_OUT="${GRAPH2VEC_OUT}/run.log" \
  HEAD_N="${HEAD_N}" \
  EVAL_N="${EVAL_N}" \
  USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
  SKIP_DATA_BUILD=true \
  bash experiments/run_graph2vec_cluster_task_eval50_7b.sh 2>&1 | tee -a "${GRAPH2VEC_OUT}/run.log"
else
  echo "==> [2/2] SKIP_GRAPH2VEC=true"
fi

echo "============================================================"
echo "Done."
echo "Data report: ${DATA_OUT%.jsonl}.selection.json"
echo "Single summary: ${SINGLE_OUT}/task_eval_summary.json"
echo "Graph2Vec summary: ${GRAPH2VEC_OUT}/eval_summary.json"
echo "Pipeline log: ${PIPELINE_LOG}"
echo "============================================================"
