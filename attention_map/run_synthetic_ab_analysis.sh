#!/usr/bin/env bash
# A-A / B-B / A-B similarity experiment on synthetic reasoning vs translation samples.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/ubuntu/work/attention_map/outputs_synthetic_ab}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/home/ubuntu/work/attention_map/analysis/synthetic_ab_experiment_k95}"

SAMPLE_A_DIR="${SAMPLE_A_DIR:-${OUTPUT_BASE}/synthetic_reasoning/logic_puzzle/sample_syn_reasoning_001_line0000}"
SAMPLE_B_DIR="${SAMPLE_B_DIR:-${OUTPUT_BASE}/synthetic_translation/en_zh_translation/sample_syn_translation_001_line0001}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate attentionmap 2>/dev/null || true
fi

# Similarity uses only the last query row (last token attention).
# Omit --no_heatmaps to generate M_*_directional.png and M_*_binary.png
python scripts/run_ab_similarity_experiment.py \
  --sample_a_dir "${SAMPLE_A_DIR}" \
  --sample_b_dir "${SAMPLE_B_DIR}" \
  --out_dir "${ANALYSIS_DIR}" \
  "$@"

python scripts/plot_per_head_sparsity.py \
  --json_path "${ANALYSIS_DIR}/per_head_sparsity.json" \
  --out_dir "${ANALYSIS_DIR}"

# Selected-token ratio: when keeping 95% of attention mass, what fraction of
# all tokens must be selected (per head + aggregate stats)?
python scripts/plot_selected_token_ratio.py \
  --json_path "${ANALYSIS_DIR}/per_head_sparsity.json" \
  --out_dir "${ANALYSIS_DIR}"

python scripts/export_ab_sample_prompts.py \
  --jsonl_path /home/ubuntu/work/datasets/synthetic_ab_pair/synthetic_ab_pair.jsonl \
  --export_dir "${ANALYSIS_DIR}/inputs" \
  --repo_root "$(pwd)" \
  --sample_a_id syn_reasoning_001 --sample_a_line 0 \
  --sample_b_id syn_translation_001 --sample_b_line 1

echo "Analysis written to ${ANALYSIS_DIR}"
