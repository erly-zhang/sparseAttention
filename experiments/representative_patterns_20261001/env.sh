#!/bin/bash
# Reuse the validated historical runtime without installing or replacing packages.
REP_VERTICAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REP_VERTICAL_ROOT/../psc_kl_quality_retry02_20260927/env.sh"
export PYTHONPATH="$REP_VERTICAL_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
