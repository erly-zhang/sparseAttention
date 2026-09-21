#!/bin/bash -l
# =============================================================================
# qhead_shared_topp  --  MATH-500 dataset-sharded evaluation (Slurm job array)
#
# One array task = ONE shard = one complete Qwen3-4B on ONE A100.
# Round-robin sharding: shard i processes samples with  idx % num_shards == i.
# All shards together cover exactly the 500 MATH-500 sample ids once.
#
# Submit (N is configurable without changing experiment semantics):
#   # N=2
#   sbatch --array=0-1 qhead_topp_array.sbatch 0.9 2
#   # N=4
#   sbatch --array=0-3 qhead_topp_array.sbatch 0.9 4
#   # N=8
#   sbatch --array=0-7 qhead_topp_array.sbatch 0.9 8
#
# After all tasks finish, merge + validate + grade:
#   python scripts/merge_shards.py --base <results>/qhead_shared_topp_p<P>/rep_seed_<RS> \
#       --out <results>/qhead_shared_topp_p<P>/rep_seed_<RS>/merged \
#       --num_shards N --expected_samples 500 --expected_runs 2
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-12:00:00          # override with --time=2-00:00:00 for p=0.95

set -euo pipefail

P="${1:-0.9}"
N="${2:-4}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=42
REP_SEED=0
TOKEN_BUDGET=1024
RESIDUAL_LENGTH=0

module purge
module load wulver

source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
conda activate random-attention

EXP_ROOT="/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918"
HARNESS="$EXP_ROOT/ra_harness"
RESULTS_ROOT="$EXP_ROOT/results"
METHOD_BASE="$RESULTS_ROOT/formal_protocolB/qhead_shared_topp/p${P}/rep_seed${REP_SEED}/sharded${N}"
OUT_DIR="$METHOD_BASE/shard_${SHARD_ID}/math_bs2_budget=${TOKEN_BUDGET}_qhead_shared_topp"
QHEAD_LOG="/tmp/qhead_topp_p${P}.s${SHARD_ID}.${SLURM_JOB_ID}.jsonl"

mkdir -p "$OUT_DIR"

export RA_DATA_DIR="/project/yl258/jz648/Random-Attention/data"
export PYTHONPATH="$HARNESS"

echo "=================================================================="
echo " qhead_shared_topp  p=$P  shard=$SHARD_ID/$N  seed=$SEED  rep_seed=$REP_SEED"
echo " job=${SLURM_JOB_ID}  array_task=${SLURM_ARRAY_TASK_ID:-0}  node=$(hostname)"
echo " out=$OUT_DIR"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HARNESS"

QHEAD_DEBUG=1 QHEAD_LOG="$QHEAD_LOG" \
python kvcompress/harness/eval_hf.py \
  --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
  --data_name math \
  --surround_with_messages \
  --attention_implementation flash_attention_2 \
  --sparsity_method eviction \
  --eviction_mode qhead_shared_topp \
  --token_budget "$TOKEN_BUDGET" \
  --residual_length "$RESIDUAL_LENGTH" \
  --top_p "$P" \
  --rep_seed "$REP_SEED" \
  --seed "$SEED" \
  --batch_size 2 \
  --num_shards "$N" \
  --shard_id "$SHARD_ID" \
  --use_batch_exist \
  --output_dir "$OUT_DIR"

# Persist raw per-selection telemetry for the merge/aggregation step.
if [ -f "$QHEAD_LOG" ]; then
  cp "$QHEAD_LOG" "$OUT_DIR/qhead_selections.jsonl"
fi

echo "=================================================================="
echo " shard $SHARD_ID done; selections -> $OUT_DIR/qhead_selections.jsonl"
echo "=================================================================="
