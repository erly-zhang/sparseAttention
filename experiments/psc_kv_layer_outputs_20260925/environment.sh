module load gcc/13.3.1-p20240614 cuda/12.4.0
export LD_LIBRARY_PATH=/jet/packages/python/v3.12.13-gcc13/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
