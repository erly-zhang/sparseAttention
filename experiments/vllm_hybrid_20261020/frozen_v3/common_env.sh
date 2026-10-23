# v3 environment (row-remap fix + our SnapKV/R-KV/VaSE port). Sourced by every v2 job.
module purge; module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh; conda activate /scratch/yl258/jz648/triattn_vllm/venv
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STUDY=/mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020
FROZEN=$STUDY/frozen_v3
export VLLM_LOGGING_CONFIG_PATH=$STUDY/logcfg_full.json
TRI_FIX=/scratch/yl258/jz648/triattn_vllm/tri_fix
TRI_HYBRID_FIX=/scratch/yl258/jz648/triattn_vllm/tri_hybrid_fix
TRI_BASELINES_FIX=/scratch/yl258/jz648/triattn_vllm/tri_baselines_fix
STATS=/mmfs1/project/yl258/jz648/Random-Attention/kvcompress/harness/stats_pl/qwen3-4b.pt
BASE64="TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0"
# arm_env <arm> <K>   (dense | rp | tri | hybrid); FIX=0 reproduces the v1 row-index defect (diagnostics only)
arm_env() {
  local fix="TRIATTN_FIX_ROW_REMAP=${FIX:-1} TRIATTN_STABLE_REQ_SEED=1"
  case $1 in
    dense) echo "PYTHONPATH=$TRI_FIX ENABLE_TRIATTENTION=0" ;;
    rp) echo "PYTHONPATH=$TRI_FIX ENABLE_TRIATTENTION=1 $fix TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=$2 $BASE64" ;;
    tri) echo "PYTHONPATH=$TRI_FIX ENABLE_TRIATTENTION=1 $fix TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS TRIATTN_RUNTIME_KV_BUDGET=$2 $BASE64" ;;
    hybrid) echo "PYTHONPATH=$TRI_HYBRID_FIX ENABLE_TRIATTENTION=1 $fix TRIATTN_RUNTIME_SELECTOR=hybrid HYBRID_K=$2 HYBRID_OVERFLOW_POLICY=raise TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=$(( $2 + 4096 )) TRIATTN_RUNTIME_DIVIDE_LENGTH=1024 TRIATTN_RUNTIME_WINDOW_SIZE=64 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0" ;;
    snapkv|rkv|vase) echo "PYTHONPATH=$TRI_BASELINES_FIX ENABLE_TRIATTENTION=1 $fix TRIATTN_RUNTIME_SELECTOR=$1 TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=$2 TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=64 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0" ;;
  esac
}
check_log() {  # check_log <log> <arm>
  local a=$(grep -c 'compression applied' $1) n=$(grep -c no_compactable_groups $1)
  local probe=$(grep -c 'TRIATTN-PROBE' $1) cap=$(grep -c 'capacity violation' $1) ovf=$(grep -c 'hybrid overflow' $1)
  echo "arm=$2 applied=$a noop=$n probe_hits=$probe capacity_violation_lines=$cap hybrid_overflow=$ovf patched_hybrid=$(grep -c 'FlashAttentionImpl.forward patched' $1) $(grep 'TRIATTN-ROWCHECK\] stats' $1 | tail -1 | sed 's/.*stats //')"
}
verify_frozen() {  # verify_frozen <out_dir>: abort unless every frozen file matches the manifest
  python3 $FROZEN/verify.py $FROZEN/manifest.json > $1/frozen_check.txt 2>&1 || { cat $1/frozen_check.txt; exit 3; }
  cp $FROZEN/manifest.json $1/frozen_manifest.json
}
