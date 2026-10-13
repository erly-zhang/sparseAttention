#!/bin/bash
#SBATCH --job-name=sparsity-monitor
#SBATCH --partition=RM-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3900M
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/baseline_sparsity_20261007/slurm-monitor-%j.out
set -euo pipefail
exec /usr/bin/python3 -u /ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/baseline_sparsity_20261007/supervisor.py
