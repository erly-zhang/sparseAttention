#!/bin/bash
set -euo pipefail
ROOT="${1:?Frozen official random_pp root required}"
PY=/project/yl258/jz648/miniconda3/envs/random-attention/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false QHEAD_DEBUG=0
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data PYTHONPATH="$ROOT/source/ra_harness"
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE LOG_UNION EVICT_TIMING
OUT="$ROOT/smoke"
mkdir "$OUT"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$OUT/exit.json"' EXIT
"$PY" - "$ROOT" "$OUT" <<'PY'
from pathlib import Path
import hashlib,json,os,sys,torch
root,out=map(Path,sys.argv[1:3]);cfg=json.loads((root/'config.json').read_text())
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
assert cfg['method']=='official_random_pp' and cfg['token_budget']==1024 and cfg['residual_length']==64
assert json.loads((root/'correctness.json').read_text())['status']=='passed'
for rel,h in cfg['model_file_hashes'].items():assert hashlib.sha256((Path(cfg['model'])/rel).read_bytes()).hexdigest()==h
assert hashlib.sha256(Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl').read_bytes()).hexdigest()==cfg['dataset_sha256']
assert os.environ.get('SLURM_JOB_ID')
p=torch.cuda.get_device_properties(0)
(out/'hardware.json').write_text(json.dumps(dict(node=os.uname().nodename,gpu_name=p.name,gpu_total_memory=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda,job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
PY
"$PY" -u "$ROOT/source/ra_harness/kvcompress/harness/eval_hf.py" \
 --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
 --data_name math --split test --prompt_type qwen-instruct --surround_with_messages \
 --attention_implementation flash_attention_2 --sparsity_method eviction \
 --eviction_mode official_random_pp --token_budget 1024 --residual_length 64 \
 --mask_seed 42 --seed 42 \
 --batch_size 2 --run_id 0 --max_tokens 32768 --warmup_new_tokens 2 \
 --ex_start_i 0 --ex_end_i 3 --use_batch_exist --output_dir "$OUT" > "$OUT/runner.log" 2>&1
