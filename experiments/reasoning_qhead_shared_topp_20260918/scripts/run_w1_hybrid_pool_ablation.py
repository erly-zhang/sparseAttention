"""Hybrid W1 pool-update ablations (I: per-step removal after attention, II: 64-step compaction,
III: per-step removal before selection) on the v6 single-launch engine."""
import os
import runpy
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base/'ra_harness'), str(base/'ra_harness/kvcompress/harness')]
from kvcompress.engine.w1_hybrid_pool_ablation_cache import install
backend = os.environ.get('RA_W1_QK_FAST_BACKEND', 'packed_qk_topk')
install(backend)
print(f'[W1 hybrid pool ablation on v6] backend={backend}', flush=True)
runpy.run_path(str(base/'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
