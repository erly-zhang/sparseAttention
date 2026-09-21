#!/bin/bash -l
# =============================================================================
# Decode-time Random Q-head Shared Top-p  --  MATH-500 formal evaluation
#
# Usage:
#   sbatch [--time=...] [--job-name=...] [--output=...] \
#          qhead_topp_eval.sbatch <top_p>
#
# One job = ONE p value = 500 unique MATH-500 problems x 2 rollouts
# (batch_size=2 expands the same prompt into two independent samples).
#
# Fair-comparison controls (identical across Dense / Random-Attention /
# Independent Top-p / Q-head Shared Top-p):
#   model, MATH-500, preprocessing, prompt/chat template, generation config,
#   generation seed, grader, dtype, environment, GPU.
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
OUT_DIR="$RESULTS_ROOT/qhead_shared_topp_p${P}/math_bs2_budget=${TOKEN_BUDGET}_qhead_shared_topp"
LOG_DIR="$EXP_ROOT/logs"
QHEAD_LOG="/tmp/qhead_topp_p${P}.${SLURM_JOB_ID}.jsonl"

mkdir -p "$OUT_DIR" "$LOG_DIR"

export RA_DATA_DIR="/project/yl258/jz648/Random-Attention/data"
export PYTHONPATH="$HARNESS"

echo "=================================================================="
echo " qhead_shared_topp  p=$P  seed=$SEED  rep_seed=$REP_SEED"
echo " job=${SLURM_JOB_ID}  node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-auto}"
echo " out=$OUT_DIR"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HARNESS"

# QHEAD_DEBUG=1 -> cheap per-selection correctness asserts
#   (prompt retained / shared mask / chronological / top-p mass >= p).
# QHEAD_LOG -> per-(layer, step, batch) selection telemetry, aggregated below.
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
  --ex_start_i 0 \
  --ex_end_i 500 \
  --use_batch_exist \
  --output_dir "$OUT_DIR"

echo "=================================================================="
echo " generation finished; aggregating qhead selection telemetry"
echo "=================================================================="

python - "$QHEAD_LOG" "$OUT_DIR" <<'PY'
import json, os, sys, statistics
log, out = sys.argv[1], sys.argv[2]
if not os.path.exists(log):
    print("no QHEAD_LOG found"); sys.exit(0)
Ks, ratios, masses, heads, kvheads = [], [], [], set(), set()
n = 0
with open(log) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        Ks.append(r["selected_reasoning_K"])
        ratios.append(r["reasoning_keep_ratio"])
        masses.append(r["selected_reasoning_mass"])
        heads.add(r["representative_q_head"])
        kvheads.add(r["representative_kv_head"])
def stat(v):
    v = sorted(v)
    return dict(n=len(v), min=v[0], max=v[-1],
                mean=round(statistics.mean(v), 4),
                median=round(statistics.median(v), 4),
                p25=v[len(v)//4], p75=v[3*len(v)//4])
summary = dict(
    selections=n,
    selected_reasoning_K=stat(Ks),
    reasoning_keep_ratio=stat(ratios),
    selected_reasoning_mass=stat(masses),
    representative_q_heads=sorted(heads),
    representative_kv_heads=sorted(kvheads),
)
with open(os.path.join(out, "qhead_stats.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "=================================================================="
echo " qhead stats saved -> $OUT_DIR/qhead_stats.json"
echo " DONE"
echo "=================================================================="
