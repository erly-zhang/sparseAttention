#!/usr/bin/env bash
set -euo pipefail

gpu_id="${GPU_ID:-2}"
wait_pid="${WAIT_PID:-49000}"
work_root="${WORK_ROOT:-/home/ubuntu/work}"
output_root="${OUTPUT_ROOT:-${work_root}/experiments/outputs/infinitebench_autoblock_topp_p_matched_sparsity}"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
runner="${work_root}/experiments/run_shareprefill_ae3_infinitebench.py"
reference_root="${work_root}/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
fixed_root="${work_root}/experiments/outputs/infinitebench_shareprefill_ae3_token_block_auto_retrieval/shareprefill_ae3_token_block_auto"
scan_root="${output_root}/scan"

mkdir -p "${output_root}/logs"
exec > >(tee -a "${output_root}/logs/gpu${gpu_id}_p_scan.log") 2>&1

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
for top_p in \
    0.930 0.935 0.940 0.945 0.950 0.955 0.960 \
    0.965 0.970 0.975 0.980 0.985 0.990; do
    tag="${top_p/./}"
    echo "[$(date --iso-8601=seconds)] scanning top-p=${top_p}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
        --method shareprefill_ae3_token_block_auto_topp \
        --target_token_top_p "${top_p}" \
        --task passkey \
        --limit 1 \
        --seed 42 \
        --calibration_scope benchmark_specific \
        --output_dir "${scan_root}/p_${tag}" \
        --reference_metrics "${reference_root}/passkey/online_metrics.jsonl" \
        --record_sparsity
done

selected_file="${scan_root}/selected_p.txt"
"${python_bin}" experiments/summarize_topp_p_scan.py \
    --scan_root "${scan_root}" \
    --fixed_metrics "${fixed_root}/passkey/online_metrics.jsonl" \
    --selected_output "${selected_file}"
selected_p="$(tr -d '[:space:]' < "${selected_file}")"
selected_tag="${selected_p/./}"
selected_root="${output_root}/selected_p_${selected_tag}_3sample"

echo "[$(date --iso-8601=seconds)] selected top-p=${selected_p}"
for task in passkey number_string longbook_choice_eng; do
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${runner}" \
        --method shareprefill_ae3_token_block_auto_topp \
        --target_token_top_p "${selected_p}" \
        --task "${task}" \
        --limit 3 \
        --seed 42 \
        --calibration_scope benchmark_specific \
        --output_dir "${selected_root}" \
        --reference_metrics "${reference_root}/${task}/online_metrics.jsonl" \
        --record_sparsity
done
echo "[$(date --iso-8601=seconds)] pure top-p sparsity-matched experiment completed"
