#!/bin/bash
set -euo pipefail
ANALYSIS_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005
source "$ANALYSIS_RUN/env.sh"
cd "$ANALYSIS_RUN"
exec "$BASE/env/bin/python" -u aggregate.py
