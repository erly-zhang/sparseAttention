#!/usr/bin/env bash
# 32k comparison sweep: 4 methods × 4 mask top_p values (503 samples = 3 head + 500 eval).
#
# Methods:
#   single_cluster | graph2vec | svd_kmeans | bmm
#
# Mask top_p sweep (sparse mask only; clustering binarize_top_p fixed at BINARIZE_TOP_P):
#   0.95 | 0.9 | 0.85 | 0.8
#
# Usage (full 16-run sweep):
#   nohup bash experiments/run_comparison_sweep_32k.sh \
#     > experiments/outputs/comparison32k_sweep.log 2>&1 &
#
# Run a single cell (example):
#   METHOD=bmm TOP_P=0.9 bash experiments/run_comparison_sweep_32k.sh
#
# Optional env:
#   SKIP_DATA_BUILD=true | METHOD=graph2vec | TOP_P=0.95
#   SWEEP_ROOT, DATA_OUT, MODEL_PATH, CHUNK_SIZE, BINARIZE_TOP_P
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/attentionmap/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/work/model/Qwen2.5-7B}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-${MAX_TOTAL_TOKENS}}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
HEAD_N="${HEAD_N:-3}"
USE_ALL_RECORDS="${USE_ALL_RECORDS:-true}"
EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS:-ff,sf,fs,ss}"
BINARIZE_TOP_P="${BINARIZE_TOP_P:-0.95}"

DATA_OUT="${DATA_OUT:-/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl}"
SWEEP_ROOT="${SWEEP_ROOT:-/home/ubuntu/work/experiments/outputs/comparison32k}"

METHODS=(single_cluster graph2vec svd_kmeans bmm)
TOP_PS=(0.95 0.9 0.85 0.8)

if [[ -n "${METHOD:-}" ]]; then
  METHODS=("${METHOD}")
fi
if [[ -n "${TOP_P:-}" ]]; then
  TOP_PS=("${TOP_P}")
fi

mkdir -p "${SWEEP_ROOT}" "$(dirname "${DATA_OUT}")"

echo "============================================================"
echo "32k comparison sweep"
echo "Model:         ${MODEL_PATH}"
echo "Data:          ${DATA_OUT}"
echo "Output root:   ${SWEEP_ROOT}"
echo "Methods:       ${METHODS[*]}"
echo "Mask top_p:    ${TOP_PS[*]}"
echo "Binarize top_p (clustering only): ${BINARIZE_TOP_P}"
echo "Chunk:         ${CHUNK_SIZE}"
echo "============================================================"

if [[ "${SKIP_DATA_BUILD:-false}" != "true" ]]; then
  echo "==> Build dataset (503 = 3 head + 500 eval @ ${MAX_TOTAL_TOKENS})"
  "${PYTHON}" experiments/build_longbench_v2_32k_domain_sample.py \
    --out "${DATA_OUT}" \
    --max_total_tokens "${MAX_TOTAL_TOKENS}" \
    --margin 64 \
    --model_path "${MODEL_PATH}" \
    --head_selection_num_samples "${HEAD_N}" \
    --head_selection_domains \
      "Long In-context Learning" \
      "Single-Document QA" \
      "Code Repository Understanding" \
    --use_all_records \
    --seed 42
else
  echo "==> SKIP_DATA_BUILD=true, reusing ${DATA_OUT}"
fi

EVAL_N="$("${PYTHON}" -c "import json; print(json.load(open('${DATA_OUT%.jsonl}.selection.json'))['eval_num_samples'])")"
echo "==> Dataset ready: head=${HEAD_N}, eval=${EVAL_N}, total=$((HEAD_N + EVAL_N))"

_run_count=0
_total_count=$((${#METHODS[@]} * ${#TOP_PS[@]}))

for method in "${METHODS[@]}"; do
  for top_p in "${TOP_PS[@]}"; do
    _run_count=$((_run_count + 1))
    top_p_tag="${top_p//./}"
    run_dir="${SWEEP_ROOT}/${method}_top_p_${top_p}"
    run_log="${run_dir}/run.log"
    mkdir -p "${run_dir}"

    echo ""
    echo "============================================================"
    echo "Run ${_run_count}/${_total_count}: method=${method} mask_top_p=${top_p}"
    echo "Out: ${run_dir}"
    echo "============================================================"

  if [[ "${method}" == "single_cluster" ]]; then
    DATA_OUT="${DATA_OUT}" \
    EXP_OUT="${run_dir}" \
    HEAD_N="${HEAD_N}" \
    EVAL_N="${EVAL_N}" \
    USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
    MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
    MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
    CHUNK_SIZE="${CHUNK_SIZE}" \
    TOP_P="${top_p}" \
    EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
    SKIP_DATA_BUILD=true \
    bash experiments/run_task_eval_10samples.sh 2>&1 | tee "${run_log}"
  else
    DATA_OUT="${DATA_OUT}" \
    EXP_OUT="${run_dir}" \
    LOG_OUT="${run_log}" \
    HEAD_N="${HEAD_N}" \
    EVAL_N="${EVAL_N}" \
    USE_ALL_RECORDS="${USE_ALL_RECORDS}" \
    MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
    MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH}" \
    CHUNK_SIZE="${CHUNK_SIZE}" \
    CLUSTER_METHOD="${method}" \
    BINARIZE_TOP_P="${BINARIZE_TOP_P}" \
    TOP_P="${top_p}" \
    EVAL_MODE_COMBOS="${EVAL_MODE_COMBOS}" \
    SKIP_DATA_BUILD=true \
    bash experiments/run_graph2vec_cluster_task_eval50_7b.sh 2>&1 | tee -a "${run_log}"
  fi

    # Lightweight run manifest for downstream aggregation.
    cat > "${run_dir}/sweep_manifest.json" <<EOF
{
  "method": "${method}",
  "mask_top_p": ${top_p},
  "binarize_top_p": ${BINARIZE_TOP_P},
  "max_total_tokens": ${MAX_TOTAL_TOKENS},
  "head_selection_num_samples": ${HEAD_N},
  "eval_num_samples": ${EVAL_N},
  "data_path": "${DATA_OUT}",
  "output_dir": "${run_dir}"
}
EOF
  done
done

echo ""
echo "============================================================"
echo "Sweep complete: ${_total_count} runs under ${SWEEP_ROOT}"
echo "============================================================"
