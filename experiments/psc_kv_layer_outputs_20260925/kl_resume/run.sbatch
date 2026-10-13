#!/bin/bash
#SBATCH --job-name=psc-kl-resume
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=01:46:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/slurm-%j.out
set -euo pipefail
cd /ocean/projects/cis260309p/jzhang114/sparseAttention
source experiments/psc_kv_layer_outputs_20260925/environment.sh
export PYTHONPATH=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/repair_01/triton300:/ocean/projects/cis260309p/jzhang114/sparseAttention${PYTHONPATH:+:$PYTHONPATH}
export PSC_RUN_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/run_kl_resume
export PSC_PIPELINE_SECONDS=6250
export PSC_APPROVED_TOTAL_SECONDS=${PSC_APPROVED_TOTAL_SECONDS:?An explicitly approved cumulative budget is required}
export TRITON_CACHE_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/kl_resume/gpu-cache300
exec timeout --signal=TERM --kill-after=10s 6290s experiments/psc_kv_layer_outputs_20260925/env/bin/python -u experiments/psc_kv_layer_outputs_20260925/kl_resume/resume.py
