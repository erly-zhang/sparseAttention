#!/bin/bash
#SBATCH --job-name=own-vs-recovery
#SBATCH --partition=RM-shared
#SBATCH --account=cis260309p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3900M
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/per_head_patterns_20261008/recovery-monitor-%j.out
set -euo pipefail
exec /usr/bin/python3 -u /ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/per_head_patterns_20261008/recovery_monitor.py
