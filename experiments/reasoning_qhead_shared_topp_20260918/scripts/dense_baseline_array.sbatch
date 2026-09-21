#!/bin/bash -l
# User approved 8 x 1 A100 / 36 hours on 2026-09-20; smoke must pass first.
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-12:00:00
#SBATCH --array=0-7
#SBATCH --job-name=math-dense

set -euo pipefail
: "${DENSE_BASELINE_APPROVED:?Explicit user approval is required}"
[[ "$DENSE_BASELINE_APPROVED" == "yes" ]]
: "${SLURM_ARRAY_TASK_ID:?Run as the approved job array}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]

module purge
module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
conda activate random-attention
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EXP_ROOT=/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data
export PYTHONPATH="$EXP_ROOT/ra_harness"
cd "$EXP_ROOT"

# Atomic creation rejects accidental duplicate launches against this output.
SHARD_DIR="$EXP_ROOT/results/dense_baseline_20260920/sharded8/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$(dirname "$SHARD_DIR")"
mkdir "$SHARD_DIR"
OUT_DIR="$SHARD_DIR/math_bs2_budget=1024_dense"
mkdir "$OUT_DIR"
trap 'rc=$?; printf "{\"returncode\":%s}\n" "$rc" > "$OUT_DIR/exit.json"' EXIT
cd "$EXP_ROOT/ra_harness"
python -u kvcompress/harness/eval_hf.py \
  --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
  --data_name math --split test --prompt_type qwen-instruct \
  --surround_with_messages --attention_implementation flash_attention_2 \
  --sparsity_method dense --token_budget 1024 \
  --seed 42 --batch_size 2 --run_id 0 --max_tokens 32768 \
  --num_shards 8 --shard_id "$SLURM_ARRAY_TASK_ID" \
  --use_batch_exist --output_dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/runner.log"
