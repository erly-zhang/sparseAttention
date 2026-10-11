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
#SBATCH --array=0-15%8
#SBATCH --job-name=mask-w16-w64
set -euo pipefail
RUN_ROOT="${1:?Frozen comparison root required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Use the job array}"
[[ "$TASK_ID" =~ ^[0-9]+$ ]] && (( TASK_ID < 16 ))
if (( TASK_ID < 8 )); then WINDOW=16; else WINDOW=64; fi
SHARD=$((TASK_ID % 8))
module purge
module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
conda activate random-attention
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RA_DATA_DIR=/project/yl258/jz648/Random-Attention/data
export PYTHONPATH="$RUN_ROOT/source/ra_harness"
export QHEAD_DEBUG=0
unset QHEAD_LOG KEEPLOG KEEPLOG_POS FORCE_KEEP_RANGE
OUT="$RUN_ROOT/w${WINDOW}/shard_${SHARD}"
mkdir "$OUT"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$OUT/exit.json"' EXIT
python - "$RUN_ROOT" "$OUT" "$WINDOW" "$SHARD" <<'PY'
import hashlib,json,os,pathlib,sys,torch
root,out=map(pathlib.Path,sys.argv[1:3])
assert json.loads((root/'smoke_gate.json').read_text())['status']=='passed'
assert json.loads((root/'warmup_gate.json').read_text())['status']=='passed'
for rel,digest in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
config=json.loads((root/'config.json').read_text())
for name,digest in config['model_file_hashes'].items():
    assert hashlib.sha256((pathlib.Path(config['model'])/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256(pathlib.Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl').read_bytes()).hexdigest()==config['dataset_sha256']
props=torch.cuda.get_device_properties(0)
(out/'hardware.json').write_text(json.dumps(dict(node=os.uname().nodename,gpu_name=props.name,gpu_total_memory=props.total_memory,torch=torch.__version__,cuda=torch.version.cuda,job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ['SLURM_ARRAY_TASK_ID']),indent=2)+'\n')
(out/'config.json').write_text(json.dumps(dict(query_window=int(sys.argv[3]),shard_id=int(sys.argv[4]),num_shards=8,p=.95,seed=42,rep_seed=0,max_tokens=32768,batch_size=2,cache_semantics='full_history_temporary_mask',warmup_new_tokens=2,per_step_json=False,debug_assertions=False),indent=2)+'\n')
PY
python -u "$RUN_ROOT/source/ra_harness/kvcompress/harness/eval_hf.py" \
  --model_name_or_path /scratch/yl258/jz648/Qwen3-4B \
  --data_name math --split test --prompt_type qwen-instruct --surround_with_messages \
  --attention_implementation flash_attention_2 --sparsity_method eviction \
  --eviction_mode qhead_shared_mask_topp --token_budget 0 --residual_length 0 \
  --top_p 0.95 --query_window "$WINDOW" --eviction_interval 1 --rep_seed 0 --seed 42 \
  --batch_size 2 --run_id 0 --max_tokens 32768 --warmup_new_tokens 2 \
  --num_shards 8 --shard_id "$SHARD" --use_batch_exist --output_dir "$OUT" > "$OUT/runner.log" 2>&1
