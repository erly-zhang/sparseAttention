#!/usr/bin/env bash
# AB analysis for LongBench-v2 7B + 32k runs (two domain sample dirs).
#
# Usage:
#   bash attention_map/run_longbench_v2_analysis_7b_32k_ab.sh
#
# Override:
#   SAMPLE_A_DIR=... SAMPLE_B_DIR=... ANALYSIS_DIR=... bash attention_map/run_longbench_v2_analysis_7b_32k_ab.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTPUT_BASE="${OUTPUT_BASE:-/home/ubuntu/work/attention_map/outputs_longbench_v2_7b_32k}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/home/ubuntu/work/attention_map/analysis/longbench_v2_7b_32k_ab_k95}"

ROOT_A_DIR="${ROOT_A_DIR:-${OUTPUT_BASE}/long_in_context_learning}"
ROOT_B_DIR="${ROOT_B_DIR:-${OUTPUT_BASE}/single_document_qa}"

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

align_sample_to_seq_len() {
  local src="$1"
  local dst="$2"
  local target_len="$3"
  python - "$src" "$dst" "$target_len" <<'PY'
import json, shutil, sys
from pathlib import Path
import numpy as np

src, dst, target_len = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
for layer_dir in sorted(src.glob("layer_*")):
    out_layer = dst / layer_dir.name
    out_layer.mkdir(parents=True, exist_ok=True)
    for head_path in sorted(layer_dir.glob("head_*.npy")):
        arr = np.load(head_path)
        if arr.ndim == 2 and arr.shape[1] > target_len:
            arr = arr[:, :target_len]
        elif arr.ndim == 1 and arr.shape[0] > target_len:
            arr = arr[:target_len]
        np.save(out_layer / head_path.name, arr.astype(np.float16, copy=False))
for meta_name in ("manifest.json", "sample_meta.json"):
    p = src / meta_name
    if p.is_file():
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["seq_len"] = target_len
        obj["aligned_from"] = str(src)
        obj["aligned_to_seq_len"] = target_len
        (dst / meta_name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

SAMPLE_A_DIR="${SAMPLE_A_DIR:-$(pick_first_sample_dir "${ROOT_A_DIR}")}"
SAMPLE_B_DIR="${SAMPLE_B_DIR:-$(pick_first_sample_dir "${ROOT_B_DIR}")}"

if [[ -z "${SAMPLE_A_DIR}" || ! -d "${SAMPLE_A_DIR}" ]]; then
  echo "[ERROR] SAMPLE_A_DIR not found under ${ROOT_A_DIR}" >&2
  exit 1
fi
if [[ -z "${SAMPLE_B_DIR}" || ! -d "${SAMPLE_B_DIR}" ]]; then
  echo "[ERROR] SAMPLE_B_DIR not found under ${ROOT_B_DIR}" >&2
  exit 1
fi

seq_a="$(read_seq_len "${SAMPLE_A_DIR}")"
seq_b="$(read_seq_len "${SAMPLE_B_DIR}")"
if [[ -z "${seq_a}" || -z "${seq_b}" ]]; then
  echo "[ERROR] Could not read seq_len from sample_meta.json/manifest.json." >&2
  exit 1
fi

ANALYSIS_B_DIR="${SAMPLE_B_DIR}"
if [[ "${seq_a}" != "${seq_b}" ]]; then
  align_len=$(( seq_a < seq_b ? seq_a : seq_b ))
  echo "[WARN] seq_len mismatch A=${seq_a}, B=${seq_b}; aligning both to ${align_len}" >&2
  ALIGN_ROOT="${ANALYSIS_DIR}/_aligned_samples"
  mkdir -p "${ALIGN_ROOT}"
  align_sample_to_seq_len "${SAMPLE_A_DIR}" "${ALIGN_ROOT}/A" "${align_len}"
  align_sample_to_seq_len "${SAMPLE_B_DIR}" "${ALIGN_ROOT}/B" "${align_len}"
  SAMPLE_A_DIR="${ALIGN_ROOT}/A"
  ANALYSIS_B_DIR="${ALIGN_ROOT}/B"
fi

echo "A: ${SAMPLE_A_DIR}"
echo "B: ${ANALYSIS_B_DIR}"
echo "Out: ${ANALYSIS_DIR}"

SAMPLE_A_DIR="${SAMPLE_A_DIR}" \
SAMPLE_B_DIR="${ANALYSIS_B_DIR}" \
ANALYSIS_DIR="${ANALYSIS_DIR}" \
bash run_longbench_v2_analysis.sh "$@"
