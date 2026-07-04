#!/usr/bin/env bash
# Same analysis flow as run_synthetic_ab_analysis.sh, but on LongBench-v2
# last-token attention outputs (each head file is the [1, seq_len] last row).
#
# Produces, under ANALYSIS_DIR:
#   - A-A / B-B / A-B directional similarity matrices (+ heatmaps, binary masks)
#   - per_head_sparsity.json (K%-mass selected tokens per head)
#   - total_selected_tokens_*.png  (plot_per_head_sparsity.py)
#   - selected_token_ratio_*.png + summary (plot_selected_token_ratio.py)
#
# A and B must share the same seq_len. long_in_context_learning currently has a
# single sample, so A and B default to the SAME sample (the pipeline still runs;
# A-A == B-B == A-B). Point SAMPLE_B_DIR at a second equal-length sample to do a
# real cross-sample comparison.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/ubuntu/work/attention_map/outputs_longbench_v2}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/home/ubuntu/work/attention_map/analysis/longbench_v2_lic_k95}"

SAMPLE_A_DIR="${SAMPLE_A_DIR:-${OUTPUT_BASE}/long_in_context_learning/new_language_translation/sample_66fcffd9bb02136c067c94c5_line0000}"
SAMPLE_B_DIR="${SAMPLE_B_DIR:-${SAMPLE_A_DIR}}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate attentionmap 2>/dev/null || true
fi

# Similarity uses only the last query row (last token), which is exactly what the
# last-token outputs store. Omit --no_heatmaps to also generate M_*_*.png.
python scripts/run_ab_similarity_experiment.py \
  --sample_a_dir "${SAMPLE_A_DIR}" \
  --sample_b_dir "${SAMPLE_B_DIR}" \
  --out_dir "${ANALYSIS_DIR}" \
  "$@"

python scripts/plot_per_head_sparsity.py \
  --json_path "${ANALYSIS_DIR}/per_head_sparsity.json" \
  --out_dir "${ANALYSIS_DIR}"

# Selected-token ratio: when keeping 95% of attention mass, what fraction of all
# tokens must be selected (per head + aggregate stats)?
python scripts/plot_selected_token_ratio.py \
  --json_path "${ANALYSIS_DIR}/per_head_sparsity.json" \
  --out_dir "${ANALYSIS_DIR}"

echo "Analysis written to ${ANALYSIS_DIR}"

# Compare two different samples:
#   SAMPLE_A_DIR=.../sampleX SAMPLE_B_DIR=.../sampleY bash run_longbench_v2_analysis.sh
# Sparsity/ratio only (skip similarity matrices):
#   bash run_longbench_v2_analysis.sh --sparsity-only
