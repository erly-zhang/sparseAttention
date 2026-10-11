#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%8
#SBATCH --job-name=rand40-rec64
set -euo pipefail
RUN_ROOT="${1:?Frozen root required}"
module purge
module load wulver
bash "$RUN_ROOT/source/scripts/random_recent_full_worker.sh" "$RUN_ROOT" "${SLURM_ARRAY_TASK_ID:?}"
