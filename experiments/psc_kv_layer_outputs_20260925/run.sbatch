#!/bin/bash
#SBATCH --job-name=psc-kv-layer5
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/slurm-%j.out
set -euo pipefail
cd /ocean/projects/cis260309p/jzhang114/sparseAttention
source experiments/psc_kv_layer_outputs_20260925/environment.sh
export PYTHONPATH=/ocean/projects/cis260309p/jzhang114/sparseAttention${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/triton-cache
exec timeout --signal=TERM --kill-after=10s 3550s experiments/psc_kv_layer_outputs_20260925/env/bin/python -u experiments/psc_kv_layer_outputs_20260925/pipeline.py
