#!/usr/bin/env bash
# AB analysis for LongBench-v2 128k runs (two different sample dirs).
#
# Usage:
#   bash attention_map/run_longbench_v2_analysis_128k_ab.sh
#
# Override inputs/outputs:
#   SAMPLE_A_DIR=... SAMPLE_B_DIR=... ANALYSIS_DIR=... bash attention_map/run_longbench_v2_analysis_128k_ab.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/ubuntu/work/attention_map/outputs_longbench_v2_128k}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/home/ubuntu/work/attention_map/analysis/longbench_v2_128k_ab_k95}"

ROOT_A_DIR="${ROOT_A_DIR:-${OUTPUT_BASE}/long_in_context_learning}"
ROOT_B_DIR="${ROOT_B_DIR:-${OUTPUT_BASE}/long_structured_data_understanding}"

pick_first_sample_dir() {
  local root="$1"
  python - "$root" <<'PY'
import glob, os, sys
root=sys.argv[1]
patterns=[
  os.path.join(root, "*", "sample_*_line*"),
  os.path.join(root, "*", "*", "sample_*_line*"),
  os.path.join(root, "**", "sample_*_line*"),
]
hits=[]
for p in patterns:
  hits.extend(glob.glob(p, recursive=True))
hits=[h for h in hits if os.path.isdir(h)]
hits=sorted(set(hits))
print(hits[0] if hits else "", end="")
PY
}

read_seq_len() {
  local sample_dir="$1"
  python - "$sample_dir" <<'PY'
import json, os, sys
d=sys.argv[1]
for name in ("sample_meta.json","manifest.json"):
  p=os.path.join(d,name)
  if os.path.isfile(p):
    with open(p,"r",encoding="utf-8") as f:
      obj=json.load(f)
    v=obj.get("seq_len")
    if v is not None:
      print(int(v), end="")
      raise SystemExit(0)
print("", end="")
PY
}

SAMPLE_A_DIR="${SAMPLE_A_DIR:-$(pick_first_sample_dir "${ROOT_A_DIR}")}"
SAMPLE_B_DIR="${SAMPLE_B_DIR:-$(pick_first_sample_dir "${ROOT_B_DIR}")}"

if [[ -z "${SAMPLE_A_DIR}" || ! -d "${SAMPLE_A_DIR}" ]]; then
  echo "[ERROR] SAMPLE_A_DIR not found: ${SAMPLE_A_DIR}" >&2
  exit 1
fi
if [[ -z "${SAMPLE_B_DIR}" || ! -d "${SAMPLE_B_DIR}" ]]; then
  echo "[ERROR] SAMPLE_B_DIR not found: ${SAMPLE_B_DIR}" >&2
  exit 1
fi

echo "A: ${SAMPLE_A_DIR}"
echo "B: ${SAMPLE_B_DIR}"
echo "Out: ${ANALYSIS_DIR}"

seq_a="$(read_seq_len "${SAMPLE_A_DIR}")"
seq_b="$(read_seq_len "${SAMPLE_B_DIR}")"
if [[ -z "${seq_a}" || -z "${seq_b}" ]]; then
  echo "[ERROR] Could not read seq_len from sample_meta.json/manifest.json." >&2
  echo "  A seq_len='${seq_a}' dir=${SAMPLE_A_DIR}" >&2
  echo "  B seq_len='${seq_b}' dir=${SAMPLE_B_DIR}" >&2
  exit 1
fi
if [[ "${seq_a}" != "${seq_b}" ]]; then
  echo "[ERROR] Sequence length mismatch: A=${seq_a}, B=${seq_b}" >&2
  echo "This AB analysis requires aligned attention maps (same seq_len)." >&2
  echo "Fix: set SAMPLE_A_DIR / SAMPLE_B_DIR to two sample_* dirs with identical seq_len." >&2
  exit 1
fi

# Delegate to the existing analysis driver (supports passing extra flags via "$@").
SAMPLE_A_DIR="${SAMPLE_A_DIR}" \
SAMPLE_B_DIR="${SAMPLE_B_DIR}" \
ANALYSIS_DIR="${ANALYSIS_DIR}" \
bash run_longbench_v2_analysis.sh "$@"

