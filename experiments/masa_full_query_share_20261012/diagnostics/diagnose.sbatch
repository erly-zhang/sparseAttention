#!/bin/bash
#SBATCH --job-name=masa-parity
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/masa_full_query_share_20261012/diagnostics/slurm-%j.out
set -euo pipefail
source /ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/masa_full_query_share_20261012/env.sh
exec "$BASE/env/bin/python" -u "$MASA_EQUAL_ROOT/diagnostics/dense_parity.py"
