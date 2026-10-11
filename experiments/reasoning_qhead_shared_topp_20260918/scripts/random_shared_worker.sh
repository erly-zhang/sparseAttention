#!/bin/bash
set -euo pipefail
RUN_ROOT="${1:?Frozen root required}"
PHASE="${2:?smoke or formal}"
SHARD="${3:-0}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
export QHEAD_DEBUG=0
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE
PYTHON=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
if [[ "$PHASE" == smoke ]]; then
  OUT="$RUN_ROOT/smoke"
  EXTRA=(--ex_start_i 0 --ex_end_i 3)
elif [[ "$PHASE" == formal && "$SHARD" =~ ^[0-7]$ ]]; then
  OUT="$RUN_ROOT/random/shard_${SHARD}"
  EXTRA=(--num_shards 8 --shard_id "$SHARD")
else
  echo 'Invalid phase or shard' >&2; exit 2
fi
mkdir "$OUT"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$OUT/exit.json"' EXIT
"$PYTHON" - "$RUN_ROOT" "$OUT" "$PHASE" <<'PY'
import hashlib,json,os,sys,torch
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);phase=sys.argv[3]
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
cfg=json.loads((root/'config.json').read_text())
assert cfg['random_keep_ratio']==.4 and cfg['selection_p'] is None and cfg['query_window'] is None
assert json.loads((root/'correctness.json').read_text())['status']=='passed'
if phase=='formal':
    gate=json.loads((root/'smoke_gate.json').read_text());assert gate['status']=='passed'
    assert hashlib.sha256((root/'smoke/summary.json').read_bytes()).hexdigest()==gate['summary_sha256']
    assert json.loads((root/'smoke/summary.json').read_text())['random_keep_ratio']==.4
for name,h in cfg['model_file_hashes'].items():assert hashlib.sha256((Path(cfg['model'])/name).read_bytes()).hexdigest()==h,name
assert hashlib.sha256(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl').read_bytes()).hexdigest()==cfg['dataset_sha256']
assert os.environ.get('SLURM_JOB_ID'),'GPU work must be inside an allocation'
props=torch.cuda.get_device_properties(0)
(out/'hardware.json').write_text(json.dumps(dict(node=os.uname().nodename,gpu_name=props.name,gpu_total_memory=props.total_memory,torch=torch.__version__,cuda=torch.version.cuda,job_id=os.environ['SLURM_JOB_ID'],array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID')),indent=2)+'\n')
(out/'config.json').write_text(json.dumps(dict(method=cfg['method'],random_keep_ratio=.4,query_window=None,selection_p=None,phase=phase,mask_seed=42,rep_seed=0,seed=42),indent=2)+'\n')
PY
"$PYTHON" -u "$RUN_ROOT/source/ra_harness/kvcompress/harness/eval_hf.py" \
  --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
  --data_name math --split test --prompt_type qwen-instruct --surround_with_messages \
  --attention_implementation flash_attention_2 --sparsity_method eviction \
  --eviction_mode qhead_shared_mask_random --token_budget 0 --residual_length 0 \
  --random_keep_ratio 0.4 --mask_seed 42 --eviction_interval 1 --rep_seed 0 --seed 42 \
  --batch_size 2 --run_id 0 --max_tokens 32768 --warmup_new_tokens 2 \
  --use_batch_exist --output_dir "$OUT" "${EXTRA[@]}" > "$OUT/runner.log" 2>&1
