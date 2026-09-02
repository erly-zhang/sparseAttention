#!/usr/bin/env bash
# Sequential orchestrator: Single-Cluster -> Graph2Vec (shared dataset).
#
# Default: 32k token budget, full LongBench-v2 eval (EVAL_N=500, USE_ALL_RECORDS=true).
#
# Usage:
#   bash experiments/run_both_eval_sequential.sh
#   EVAL_N=50 bash experiments/run_both_eval_sequential.sh          # partial run
#   nohup bash experiments/run_both_eval_sequential.sh > experiments/outputs/run_both.log 2>&1 &
#
# For 128k full corpus use instead:
#   bash experiments/run_full_longbench_128k_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/attentionmap/bin/python}"

HEAD_N="${HEAD_N:-3}"
EVAL_N="${EVAL_N:-500}"
USE_ALL_RECORDS="${USE_ALL_RECORDS:-true}"
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS:-ff,sf,fs,ss}"
export EVAL_MODE_COMBOS

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-${MAX_TOTAL_TOKENS}}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"

# Shared dataset for both experiments (head + eval lines).
DATA_OUT="${DATA_OUT:-/home/ubuntu/work/experiments/data/longbench_v2_32k_full_eval${EVAL_N}_7b.jsonl}"

SINGLE_OUT="${SINGLE_OUT:-/home/ubuntu/work/experiments/outputs/shared_layer_mask_task_eval${EVAL_N}_7b}"
GRAPH2VEC_OUT="${GRAPH2VEC_OUT:-/home/ubuntu/work/experiments/outputs/graph2vec_cluster2_task_eval${EVAL_N}_7b}"

SINGLE_LOG="${SINGLE_LOG:-/home/ubuntu/work/experiments/outputs/run_single_cluster_task_eval${EVAL_N}_7b.log}"
GRAPH2VEC_LOG="${GRAPH2VEC_LOG:-/home/ubuntu/work/experiments/outputs/run_graph2vec_cluster2_task_eval${EVAL_N}_7b.log}"

mkdir -p "$(dirname "${DATA_OUT}")" "$(dirname "${SINGLE_LOG}")" "$(dirname "${GRAPH2VEC_LOG}")"

echo "============================================================"
echo "Sequential eval: Single-Cluster -> Graph2Vec"
echo "  HEAD_N=${HEAD_N}  EVAL_N=${EVAL_N}  total lines=$((HEAD_N + EVAL_N))"
echo "  DATA_OUT=${DATA_OUT}"
echo "  SINGLE_OUT=${SINGLE_OUT}"
echo "  GRAPH2VEC_OUT=${GRAPH2VEC_OUT}"
echo "============================================================"
echo ""
echo "NOTE: LongBench-v2 full corpus has ~503 records."
echo "      Max unique eval samples is ~500 (3 used for head selection)."
echo "      EVAL_N=2000 will fail unless you point --source to a larger dataset."
echo ""

echo "==> [1/2] Single-Cluster (run_task_eval_10samples.sh)"
HEAD_N="${HEAD_N}" \
EVAL_N="${EVAL_N}" \
USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
CHUNK_SIZE="${CHUNK_SIZE}" \
DATA_OUT="${DATA_OUT}" \
EXP_OUT="${SINGLE_OUT}" \
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
bash experiments/run_task_eval_10samples.sh 2>&1 | tee "${SINGLE_LOG}"

echo ""
echo "==> [2/2] Graph2Vec 2-Cluster (run_graph2vec_cluster_task_eval50_7b.sh)"
HEAD_N="${HEAD_N}" \
EVAL_N="${EVAL_N}" \
USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
CHUNK_SIZE="${CHUNK_SIZE}" \
DATA_OUT="${DATA_OUT}" \
EXP_OUT="${GRAPH2VEC_OUT}" \
LOG_OUT="${GRAPH2VEC_OUT}/run.log" \
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
SKIP_DATA_BUILD=true \
bash experiments/run_graph2vec_cluster_task_eval50_7b.sh 2>&1 | tee "${GRAPH2VEC_LOG}"

echo ""
echo "============================================================"
echo "All done."
echo "  Single summary:   ${SINGLE_OUT}/task_eval_summary.json"
echo "  Graph2Vec summary:  ${GRAPH2VEC_OUT}/eval_summary.json"
echo "  Single log:         ${SINGLE_LOG}"
echo "  Graph2Vec log:      ${GRAPH2VEC_LOG}"
echo "============================================================"
