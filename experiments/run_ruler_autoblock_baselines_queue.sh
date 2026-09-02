#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 GPU_ID LENGTH [LENGTH ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
LENGTHS=("$@")

WORK_ROOT=/home/ubuntu/work
RUNNER="$WORK_ROOT/experiments/run_shareprefill_ae3_ruler.py"
MODEL="$WORK_ROOT/model/Qwen2.5-7B"
DATA_ROOT="$WORK_ROOT/FlexPrefill/experiments/benchmark/ruler/data/qwen2_5_formal_200"
OUTPUT_ROOT="$WORK_ROOT/experiments/outputs/ruler_autoblock_baselines_base_nochat_20260817"
FLEX_PY=/home/ubuntu/miniconda3/envs/official_flex/bin/python
MI_PY=/home/ubuntu/miniconda3/envs/official_mi/bin/python
TASKS=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

mkdir -p "$OUTPUT_ROOT/logs"

validate_method() {
  local length="$1"
  local method="$2"
  local expected_alignment="$3"
  "$FLEX_PY" - "$OUTPUT_ROOT/length_$length" "$method" "$length" "$expected_alignment" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
method = sys.argv[2]
length = sys.argv[3]
expected_alignment = sys.argv[4]
method_root = root / method
metrics = method_root / "online_metrics.jsonl"
summary = method_root / "summary.json"
tasks = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
]
if not metrics.is_file() or not summary.is_file():
    raise SystemExit(f"missing metrics or summary for {method} length={length}")
metric_count = sum(1 for line in metrics.open() if line.strip())
if metric_count != 2600:
    raise SystemExit(f"metrics count={metric_count}, expected=2600")
for task in tasks:
    path = method_root / length / f"{task}.jsonl"
    count = sum(1 for line in path.open() if line.strip()) if path.is_file() else 0
    if count != 200:
        raise SystemExit(f"{task} count={count}, expected=200")
data = json.load(summary.open())
run_args = data.get("run_args", {})
if run_args.get("chat") is not False:
    raise SystemExit(f"expected base/no-chat protocol, got run_args={run_args}")
if Path(str(run_args.get("model", ""))).name != "Qwen2.5-7B":
    raise SystemExit(f"unexpected model checkpoint: {run_args.get('model')}")
protocol = data.get("method_metadata", {}).get("prompt_protocol", {})
if protocol.get("format") != "raw_completion":
    raise SystemExit(f"unexpected prompt protocol: {protocol}")
alignment = data.get("input_alignment", {})
if alignment.get("status") != expected_alignment:
    raise SystemExit(f"alignment={alignment}, expected status={expected_alignment}")
if expected_alignment == "passed" and alignment.get("alignment_scope") != "full":
    raise SystemExit(f"alignment scope is not full: {alignment}")
print(f"validated method={method} length={length} metrics={metric_count} alignment={alignment.get('status')}")
PY
}

for LENGTH in "${LENGTHS[@]}"; do
  ROOT="$OUTPUT_ROOT/length_$LENGTH"
  mkdir -p "$ROOT"
  AUTO_LOG="$OUTPUT_ROOT/logs/gpu${GPU_ID}_${LENGTH}_autoblock.log"
  FLEX_LOG="$OUTPUT_ROOT/logs/gpu${GPU_ID}_${LENGTH}_flexprefill.log"
  MI_LOG="$OUTPUT_ROOT/logs/gpu${GPU_ID}_${LENGTH}_minference.log"

  echo "[$(date -Is)] GPU=$GPU_ID length=$LENGTH method=autoblock start"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$FLEX_PY" "$RUNNER" \
    --model "$MODEL" \
    --no-chat \
    --method shareprefill_ae3_token_block_auto \
    --lengths "$LENGTH" \
    --tasks "${TASKS[@]}" \
    --data_root "$DATA_ROOT" \
    --output_dir "$ROOT" \
    --reference_run \
    --record_sparsity >>"$AUTO_LOG" 2>&1
  validate_method "$LENGTH" shareprefill_ae3_token_block_auto reference_created

  REFERENCE="$ROOT/shareprefill_ae3_token_block_auto/online_metrics.jsonl"
  echo "[$(date -Is)] GPU=$GPU_ID length=$LENGTH method=flexprefill start"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$FLEX_PY" "$RUNNER" \
    --model "$MODEL" \
    --no-chat \
    --method flexprefill \
    --lengths "$LENGTH" \
    --tasks "${TASKS[@]}" \
    --data_root "$DATA_ROOT" \
    --output_dir "$ROOT" \
    --reference_metrics "$REFERENCE" \
    --record_sparsity >>"$FLEX_LOG" 2>&1
  validate_method "$LENGTH" flexprefill passed

  echo "[$(date -Is)] GPU=$GPU_ID length=$LENGTH method=minference start"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONPATH="$WORK_ROOT/MInference:$WORK_ROOT" \
  "$MI_PY" "$RUNNER" \
    --model "$MODEL" \
    --no-chat \
    --method minference \
    --lengths "$LENGTH" \
    --tasks "${TASKS[@]}" \
    --data_root "$DATA_ROOT" \
    --output_dir "$ROOT" \
    --reference_metrics "$REFERENCE" \
    --record_sparsity \
    --evaluator_python "$FLEX_PY" >>"$MI_LOG" 2>&1
  validate_method "$LENGTH" minference passed

  echo "[$(date -Is)] GPU=$GPU_ID length=$LENGTH all_methods=complete"
done
