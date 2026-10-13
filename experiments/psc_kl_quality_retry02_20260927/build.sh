#!/bin/bash
set -euo pipefail
HERE=/jet/home/jzhang114/vscode_cli/runtimebuild
exec > "$HERE/build.log" 2>&1
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
BASE="$ROOT/experiments/psc_kv_layer_outputs_20260925"
module purge
source "$BASE/environment.sh"
GCC=/opt/packages/gcc/v11.3.0/b2rm
export LD_LIBRARY_PATH="$GCC/lib64:/jet/packages/python/v3.12.13-gcc13/lib"
unset LD_PRELOAD
export CC="$GCC/bin/gcc" CXX="$GCC/bin/g++" CUDAHOSTCXX="$GCC/bin/g++"
export COMPILER_PATH="$GCC/bin:/usr/bin"
export PATH="$GCC/bin:$PATH"
export MAX_JOBS=2 OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST='8.9;9.0'
export CFLAGS='-O2 -march=x86-64 -mtune=generic'
export CXXFLAGS="$CFLAGS"
export MINFERENCE_FORCE_BUILD=TRUE
"$CXX" -x c++ -fsyntax-only /dev/null
"$BASE/env/bin/python" -u -c 'import torch; print(torch.__version__, flush=True)'
timeout --signal=TERM --kill-after=15s 900s "$BASE/env/bin/python" -m pip wheel \
  --no-deps --no-build-isolation --no-binary=minference --no-cache-dir \
  --wheel-dir "$HERE/wheels" minference==0.1.6.0
