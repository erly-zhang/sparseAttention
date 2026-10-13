ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
BASE="$ROOT/experiments/psc_kv_layer_outputs_20260925"
module purge
source "$BASE/environment.sh"
export LD_LIBRARY_PATH=/jet/packages/python/v3.12.13-gcc13/lib
export LD_PRELOAD=/opt/packages/gcc/v11.3.0/b2rm/lib64/libstdc++.so.6
export CC=/usr/bin/gcc
export COMPILER_PATH=/usr/bin
RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$RUNTIME_DIR/overlay:$BASE/repair_01/triton300:$BASE:$ROOT"
