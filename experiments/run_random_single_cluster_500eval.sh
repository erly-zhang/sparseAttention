#!/usr/bin/env bash
# Experiment 1: Random single-cluster representative head baseline.
# TOP_P=0.85, 500 eval, ff/sf/fs/ss, seed=42.
#
# Usage:
#   bash experiments/run_random_single_cluster_500eval.sh
#   nohup bash experiments/run_random_single_cluster_500eval.sh \
#     > experiments/outputs/random_single_cluster_top_p_0.85_500eval/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

BASELINE_ID=random_single_cluster \
EXP_OUT=/home/ubuntu/work/experiments/outputs/random_single_cluster_top_p_0.85_500eval \
LOG_OUT=/home/ubuntu/work/experiments/outputs/random_single_cluster_top_p_0.85_500eval/run.log \
EVAL_N=500 \
USE_ALL_RECORDS=false \
SEED=42 \
EVAL_MODE_COMBOS=ff,sf,fs,ss \
SKIP_DATA_BUILD="${SKIP_DATA_BUILD:-false}" \
bash experiments/run_baseline_eval.sh
