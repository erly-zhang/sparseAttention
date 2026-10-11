"""Opt in to the exact full-KV W1 implementation, then run the usual runner.

Usage: PYTHONPATH=.../ra_harness python run_full_w1_optimized.py [eval_hf args]
RA_FULL_W1_RANK_BACKEND defaults to stable_sort; packed_topk is also exact.
"""
import os
import runpy
from pathlib import Path
import sys

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base / 'ra_harness'), str(base / 'ra_harness/kvcompress/harness')]
from kvcompress.engine.full_w1_optimized_cache import install

backend = os.environ.get('RA_FULL_W1_RANK_BACKEND', 'stable_sort')
install(backend)
print(f'[full-W1 implementation] full_w1_fixed_ranges_batched_exact_v1 rank_backend={backend}', flush=True)
runpy.run_path(str(base / 'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
