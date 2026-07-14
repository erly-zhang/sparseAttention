#!/usr/bin/env bash
# Full LongBench-v2 pipeline (defaults to 32k for quick testing):
# build data -> Single-Cluster -> Graph2Vec-style cluster runner.
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
  "${PYTHON}" experiments/run_shared_layer_mask_experiment.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_OUT}" \
    --output_dir "${SINGLE_OUT}" \
    --num_samples "${TOTAL_N}" \
    --head_selection_num_samples "${HEAD_N}" \
    --eval_num_samples "${EVAL_N}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --last_q 32 \
    --chunk_size "${CHUNK_SIZE}" \
    --dtype bf16 \
    --device cuda \
    --seed 42 \
    --mask_method top_p \
    --top_p "${TOP_P:-0.85}" \
    --top_k 128 \
    --local_window 256 \
    --representative_selection coverage \
    --run_task_eval true \
    --eval_mode_combos "${EVAL_MODE_COMBOS}" \
    --eval_max_new_tokens 8 \
    --eval_compute_ppl true \
    --do_sample false \
    --temperature 1.0 \
    --save_masks false \
    --save_similarity true \
    --filter_after_run false 2>&1 | tee "${SINGLE_OUT}/run.log"
else
  echo "==> [1/2] SKIP_SINGLE_CLUSTER=true"
fi

if [[ "${SKIP_GRAPH2VEC:-false}" != "true" ]]; then
  echo "==> [2/2] Graph2Vec 2-Cluster shared mask eval"
  "${PYTHON}" experiments/run_graph2vec_cluster_shared_mask_experiment.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_OUT}" \
    --output_dir "${GRAPH2VEC_OUT}" \
    --num_samples "${TOTAL_N}" \
    --head_selection_num_samples "${HEAD_N}" \
    --eval_num_samples "${EVAL_N}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --last_q 32 \
    --chunk_size "${CHUNK_SIZE}" \
    --dtype bf16 \
    --device cuda \
    --seed 42 \
    --mask_method top_p \
    --top_p "${TOP_P:-0.85}" \
    --top_k 128 \
    --local_window 256 \
    --run_task_eval true \
    --eval_mode_combos "${EVAL_MODE_COMBOS}" \
    --eval_max_new_tokens 8 \
    --eval_compute_ppl true \
    --do_sample false \
    --temperature 1.0 \
    --save_masks false \
    --save_similarity true \
    --filter_after_run false \
    --cluster_method "${CLUSTER_METHOD:-graph2vec}" \
    --num_head_clusters 2 \
    --binarize_method top_p \
    --binarize_top_p "${BINARIZE_TOP_P:-0.95}" \
    --binarize_top_k 128 \
    --graph_type bipartite \
    --graph2vec_dim 128 \
    --graph2vec_wl_iterations 2 \
    --graph2vec_workers 1 \
    --cluster_seed 42 \
    --svd_components 8 \
    --bmm_max_iter 100 \
    --bmm_tol 0.0001 \
    --bmm_n_init 5 \
    --sink_tokens 4 \
    --save_graph2vec_embeddings true 2>&1 | tee -a "${GRAPH2VEC_OUT}/run.log"
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
