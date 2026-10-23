#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=vllm-setup
set -euo pipefail
module purge; module load wulver
nvidia-smi; nvidia-smi --query-gpu=driver_version,name --format=csv
WORK=/scratch/yl258/jz648/triattn_vllm
export RA_ROOT=/mmfs1/project/yl258/jz648/Random-Attention
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh
[ -x $WORK/venv/bin/python ] || conda create -y -p $WORK/venv python=3.12 pip
source $WORK/venv/bin/activate 2>/dev/null || conda activate $WORK/venv
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=$WORK/pip_cache
python -V
pip install -U pip
pip install vllm==0.19.0
[ -d $WORK/triattention_src ] || git clone https://github.com/WeianMao/triattention $WORK/triattention_src
git -C $WORK/triattention_src checkout 9c26d14
git -C $WORK/triattention_src log -1 --format='%H %ad'
cd $WORK
rsync -a --exclude .git triattention_src/ tri/
cp $RA_ROOT/scripts/vllm_rp_bench/selector_random.py tri/triattention/vllm/runtime/
cp $RA_ROOT/scripts/vllm_rp_bench/shims/*.py tri/triattention/vllm/runtime/
cd tri && python $RA_ROOT/scripts/vllm_rp_bench/apply_hook_edit.py triattention/vllm/runtime/hook_impl.py
pip install -e . --no-deps
python - <<'PY'
import vllm, torch, triton
from importlib.metadata import entry_points
print('vllm', vllm.__version__, 'torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda)
print('plugins', [e.name for e in entry_points(group="vllm.general_plugins")])
PY
pip freeze > $WORK/pip_freeze.txt
# Smoke: random_pp arm, short run, Qwen3-4B local checkpoint.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
env ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 \
    TRIATTN_RUNTIME_KV_BUDGET=1024 TRIATTN_RUNTIME_DIVIDE_LENGTH=128 TRIATTN_RUNTIME_WINDOW_SIZE=128 \
    TRIATTN_RUNTIME_PROTECT_PREFILL=1 \
    vllm bench latency --model /scratch/yl258/jz648/Qwen3-4B --dtype bfloat16 --input-len 256 --output-len 2048 \
      --batch-size 4 --num-iters 1 --num-iters-warmup 1 --max-model-len 4096 --enforce-eager --no-enable-prefix-caching \
      2>&1 | tee $WORK/smoke_random_pp.log | grep -E "TriAttention|status=applied|Avg latency|Error|error" | head -40
echo SETUP_DONE
