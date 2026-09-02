#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 GPU_ID START_LAYER [START_LAYER ...]" >&2
    exit 2
fi

gpu_id="$1"
shift
limit="${LIMIT:-3}"
work_root="${WORK_ROOT:-/home/ubuntu/work}"
scan_root="${SCAN_ROOT:-${work_root}/experiments/outputs/infinitebench_dense_topp_mass_layer_scan_20260818}"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
runner="${work_root}/experiments/run_shareprefill_ae3_infinitebench.py"
reference_root="${work_root}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
method="shareprefill_ae3_token_block_auto_dense_topp_mass"

cd "${work_root}"
for start_layer in "$@"; do
    output_root="${scan_root}/start_${start_layer}"
    if [[ -e "${output_root}" ]]; then
        echo "Refusing to overwrite existing output: ${output_root}" >&2
        exit 1
    fi
    mkdir -p "${output_root}/logs"
    log_path="${output_root}/logs/gpu${gpu_id}.log"
    for task in passkey number_string; do
        reference_metrics="${reference_root}/${task}/online_metrics.jsonl"
        echo "[$(date --iso-8601=seconds)] start_layer=${start_layer} starting ${task}" | tee -a "${log_path}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
            --method "${method}" \
            --target_top_p_start_layer "${start_layer}" \
            --target_token_top_p 0.95 \
            --task "${task}" \
            --limit "${limit}" \
            --seed 42 \
            --calibration_scope benchmark_specific \
            --output_dir "${output_root}" \
            --reference_metrics "${reference_metrics}" \
            --record_sparsity 2>&1 | tee -a "${log_path}"
        echo "[$(date --iso-8601=seconds)] start_layer=${start_layer} completed ${task}" | tee -a "${log_path}"
    done
done
