#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?usage: $0 GPU RATIO [RATIO ...]}
shift

WORK_ROOT=${WORK_ROOT:-/home/ubuntu/work}
LIMIT=${LIMIT:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/experiments/outputs/infinitebench_autoblock_topp_block_threshold_smoke10}
REFERENCE=${REFERENCE:-${WORK_ROOT}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full/passkey/online_metrics.jsonl}
PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}

cd "${WORK_ROOT}"
for ratio in "$@"; do
  tag=$(printf "%03d" "$(awk -v r="${ratio}" 'BEGIN { print int(r * 100 + 0.5) }')")
  output_dir="${OUTPUT_ROOT}/threshold_${tag}pct"
  echo "[$(date -Is)] GPU${GPU}: threshold=${ratio}, limit=${LIMIT}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    experiments/run_shareprefill_ae3_infinitebench.py \
    --method shareprefill_ae3_token_block_auto_topp \
    --task passkey \
    --limit "${LIMIT}" \
    --seed 42 \
    --calibration_scope benchmark_specific \
    --minimum_block_target_probability_ratio "${ratio}" \
    --reference_metrics "${REFERENCE}" \
    --output_dir "${output_dir}" \
    --record_sparsity
done
