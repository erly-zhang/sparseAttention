#!/usr/bin/env bash
set -euo pipefail

gpu_id="${GPU_ID:-2}"
wait_pid="${WAIT_PID:-49000}"
limit="${LIMIT:-3}"
work_root="${WORK_ROOT:-/home/ubuntu/work}"
output_root="${OUTPUT_ROOT:-${work_root}/experiments/outputs/infinitebench_autoblock_topp90_matched8192_3sample}"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
runner="${work_root}/experiments/run_shareprefill_ae3_infinitebench.py"
reference_root="${work_root}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"

mkdir -p "${output_root}/logs"
log_file="${output_root}/logs/gpu${gpu_id}_matched_smoke.log"
exec > >(tee -a "${log_file}") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for queue PID ${wait_pid}"
while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 60
done

echo "[$(date --iso-8601=seconds)] queue exited; waiting for GPU ${gpu_id}"
while nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits \
    --id="${gpu_id}" | grep -q '[0-9]'; do
    sleep 60
done

cd "${work_root}"
for task in passkey number_string longbook_choice_eng; do
    reference_metrics="${reference_root}/${task}/online_metrics.jsonl"
    if [[ ! -f "${reference_metrics}" ]]; then
        echo "missing reference metrics: ${reference_metrics}" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] GPU ${gpu_id} starting ${task}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
        --method shareprefill_ae3_token_block_auto_topp_matched \
        --task "${task}" \
        --limit "${limit}" \
        --seed 42 \
        --calibration_scope benchmark_specific \
        --output_dir "${output_root}" \
        --reference_metrics "${reference_metrics}" \
        --record_sparsity
    echo "[$(date --iso-8601=seconds)] GPU ${gpu_id} completed ${task}"
done

echo "[$(date --iso-8601=seconds)] matched-budget smoke queue completed"
