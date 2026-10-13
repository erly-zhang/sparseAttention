#!/bin/bash
#SBATCH --job-name=jlens-supervisor
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3900M
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005/supervision/slurm-supervisor-%j.out
set -euo pipefail
ANALYSIS_RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/jlens_prefill_analysis_20261005
cd "$ANALYSIS_RUN/supervision"
exec /usr/bin/python3 -u controller_entry.py
