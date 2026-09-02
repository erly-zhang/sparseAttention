#!/usr/bin/env bash
# Graph2Vec 2-Cluster shared sparse mask experiment (two-phase task eval).
#
# Phase 1: Graph2Vec head clustering + cluster representatives
# Phase 2: task eval with fixed global cluster masks (ff/sf/fs/ss)
#
# Override via env:
#   EVAL_N=50 | USE_ALL_RECORDS=true | MAX_TOTAL_TOKENS=131072 | SKIP_DATA_BUILD=true
#   DEBUG_LAYERS="0,1"     # smoke test on selected layers only
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
EXP_OUT="${EXP_OUT:-/home/ubuntu/work/experiments/outputs/graph2vec_cluster2_task_eval50_7b}"
LOG_OUT="${LOG_OUT:-${EXP_OUT}/run.log}"

HEAD_N="${HEAD_N:-3}"
EVAL_N="${EVAL_N:-50}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-${MAX_TOTAL_TOKENS}}"
TOKEN_MARGIN="${TOKEN_MARGIN:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
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

# Graph2Vec / clustering knobs (override via env if needed).
NUM_HEAD_CLUSTERS="${NUM_HEAD_CLUSTERS:-2}"
BINARIZE_METHOD="${BINARIZE_METHOD:-top_p}"
BINARIZE_TOP_P="${BINARIZE_TOP_P:-0.95}"
GRAPH_TYPE="${GRAPH_TYPE:-bipartite}"
GRAPH2VEC_DIM="${GRAPH2VEC_DIM:-128}"
GRAPH2VEC_WL_ITERATIONS="${GRAPH2VEC_WL_ITERATIONS:-2}"
CLUSTER_METHOD="${CLUSTER_METHOD:-graph2vec}"
TOP_P="${TOP_P:-0.95}"
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS:-ff,sf,fs,ss}"
DEBUG_LAYERS="${DEBUG_LAYERS:-}"  # e.g. "0,1" for smoke test only

echo "==> Model: ${MODEL_PATH}"
echo "==> Data:  ${DATA_OUT}"
echo "==> Out:   ${EXP_OUT}"
echo "==> Log:   ${LOG_OUT}"
echo "==> Head selection: ${HEAD_N} | Eval: ${EVAL_N} | Clusters/layer: ${NUM_HEAD_CLUSTERS} | Method: ${CLUSTER_METHOD} | mask top_p: ${TOP_P}"

mkdir -p "$(dirname "${EXP_OUT}")" "$(dirname "${LOG_OUT}")"

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

echo "==> Step 2: ${CLUSTER_METHOD} cluster head selection (${HEAD_N}) + cluster sparse eval (${EVAL_N})"

GRAPH2VEC_ARGS=(
  experiments/run_graph2vec_cluster_shared_mask_experiment.py
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_OUT}"
  --output_dir "${EXP_OUT}"
  --num_samples "$((HEAD_N + EVAL_N))"
  --head_selection_num_samples "${HEAD_N}"
  --eval_num_samples "${EVAL_N}"
  --skip_head_selection false
  --max_input_length "${MAX_INPUT_LENGTH}"
  --last_q 32
  --chunk_size "${CHUNK_SIZE}"
  --cluster_method "${CLUSTER_METHOD}"
  --num_head_clusters "${NUM_HEAD_CLUSTERS}"
  --binarize_method "${BINARIZE_METHOD}"
  --binarize_top_p "${BINARIZE_TOP_P}"
  --graph_type "${GRAPH_TYPE}"
  --graph2vec_dim "${GRAPH2VEC_DIM}"
  --graph2vec_wl_iterations "${GRAPH2VEC_WL_ITERATIONS}"
  --mask_method top_p
  --top_p "${TOP_P}"
  --run_task_eval true
  --eval_mode_combos "${EVAL_MODE_COMBOS}"
  --eval_max_new_tokens 8
  --save_masks false
  --save_similarity true
  --save_graph2vec_embeddings true
  --filter_after_run false
)

if [[ -n "${DEBUG_LAYERS}" ]]; then
  GRAPH2VEC_ARGS+=(--debug_layers "${DEBUG_LAYERS}")
fi

"${PYTHON}" "${GRAPH2VEC_ARGS[@]}" 2>&1 | tee "${LOG_OUT}"

echo "==> Done."
echo "Selection report: ${DATA_OUT%.jsonl}.selection.json"
echo "Global clusters: ${EXP_OUT}/global_cluster_assignments.json"
echo "Global representatives: ${EXP_OUT}/global_cluster_representatives.json"
echo "Head selection summary: ${EXP_OUT}/head_selection_summary.json"
echo "Eval summary: ${EXP_OUT}/eval_summary.json"
echo "Run log: ${LOG_OUT}"
