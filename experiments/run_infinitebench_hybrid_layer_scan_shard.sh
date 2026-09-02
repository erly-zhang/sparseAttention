#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 GPU_ID TOP_P_START_LAYER" >&2
    exit 2
fi

gpu_id="$1"
start_layer="$2"
limit="${LIMIT:-3}"
work_root="${WORK_ROOT:-/home/ubuntu/work}"
scan_root="${SCAN_ROOT:-${work_root}/experiments/outputs/infinitebench_autoblock_hybrid_layer_scan_20260818}"
output_root="${scan_root}/start_${start_layer}"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
runner="${work_root}/experiments/run_shareprefill_ae3_infinitebench.py"
reference_root="${work_root}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
method="shareprefill_ae3_token_block_auto_hybrid"

if [[ -e "${output_root}" ]]; then
    echo "Refusing to overwrite existing output: ${output_root}" >&2
    exit 1
fi
mkdir -p "${output_root}/logs"
exec > >(tee -a "${output_root}/logs/gpu${gpu_id}.log") 2>&1

cd "${work_root}"
for task in passkey number_string; do
    reference_metrics="${reference_root}/${task}/online_metrics.jsonl"
    if [[ ! -f "${reference_metrics}" ]]; then
        echo "Missing reference metrics: ${reference_metrics}" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] start_layer=${start_layer} starting ${task}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
        --method "${method}" \
        --target_top_p_start_layer "${start_layer}" \
        --target_token_top_p 0.90 \
        --task "${task}" \
        --limit "${limit}" \
        --seed 42 \
        --calibration_scope benchmark_specific \
        --output_dir "${output_root}" \
        --reference_metrics "${reference_metrics}" \
        --record_sparsity
    echo "[$(date --iso-8601=seconds)] start_layer=${start_layer} completed ${task}"
done

echo "[$(date --iso-8601=seconds)] start_layer=${start_layer} shard completed"
