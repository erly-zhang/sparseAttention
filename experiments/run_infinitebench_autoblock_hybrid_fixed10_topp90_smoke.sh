#!/usr/bin/env bash
set -euo pipefail

gpu_id="${GPU_ID:-2}"
limit="${LIMIT:-3}"
work_root="${WORK_ROOT:-/home/ubuntu/work}"
output_root="${OUTPUT_ROOT:-${work_root}/experiments/outputs/infinitebench_autoblock_hybrid_fixed10_topp90_smoke_v2_20260818}"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
runner="${work_root}/experiments/run_shareprefill_ae3_infinitebench.py"
reference_root="${work_root}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
method="shareprefill_ae3_token_block_auto_hybrid_fixed10_topp90"

if [[ -e "${output_root}" ]]; then
    echo "Refusing to overwrite existing output: ${output_root}" >&2
    exit 1
fi
mkdir -p "${output_root}/logs"
exec > >(tee -a "${output_root}/logs/gpu${gpu_id}.log") 2>&1

cd "${work_root}"
for task in passkey number_string longbook_choice_eng; do
    reference_metrics="${reference_root}/${task}/online_metrics.jsonl"
    if [[ ! -f "${reference_metrics}" ]]; then
        echo "Missing reference metrics: ${reference_metrics}" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] starting ${task} (${limit} samples)"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
        --method "${method}" \
        --task "${task}" \
        --limit "${limit}" \
        --seed 42 \
        --calibration_scope benchmark_specific \
        --output_dir "${output_root}" \
        --reference_metrics "${reference_metrics}" \
        --record_sparsity
    echo "[$(date --iso-8601=seconds)] completed ${task}"
done

echo "[$(date --iso-8601=seconds)] hybrid smoke completed"
