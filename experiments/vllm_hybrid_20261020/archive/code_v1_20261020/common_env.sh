# Sourced by every vLLM job of this study.
module purge; module load wulver
source /project/yl258/jz648/miniconda3/etc/profile.d/conda.sh; conda activate /scratch/yl258/jz648/triattn_vllm/venv
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_CONFIG_PATH=/mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020/logcfg_full.json
STUDY=/mmfs1/project/yl258/jz648/sparseAttention/experiments/vllm_hybrid_20261020
TRI_HYBRID=/scratch/yl258/jz648/triattn_vllm/tri_hybrid
# arm_env <arm> <K>: environment for one arm (dense | rp | tri | hybrid)
arm_env() {
  case $1 in
    dense) echo "ENABLE_TRIATTENTION=0" ;;
    rp) echo "ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=$2 TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0" ;;
    tri) echo "ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SPARSE_STATS_PATH=$3 TRIATTN_RUNTIME_KV_BUDGET=$2 TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0" ;;
    hybrid) echo "PYTHONPATH=$TRI_HYBRID ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=hybrid HYBRID_K=$2 TRIATTN_RUNTIME_RANDOM_SEED=1234 TRIATTN_RUNTIME_KV_BUDGET=$(( $2 + 4096 )) TRIATTN_RUNTIME_DIVIDE_LENGTH=1024 TRIATTN_RUNTIME_WINDOW_SIZE=64 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0" ;;
  esac
}
check_log() {  # check_log <log> <arm>
  local a=$(grep -c 'compression applied' $1) n=$(grep -c no_compactable_groups $1) p=$(grep -ci 'preempt' $1)
  echo "arm=$2 applied=$a noop=$n preempt_lines=$p hybrid_patched=$(grep -c 'FlashAttentionImpl.forward patched' $1) $(grep -E 'Throughput:|Avg latency' $1 | tail -1)"
}
