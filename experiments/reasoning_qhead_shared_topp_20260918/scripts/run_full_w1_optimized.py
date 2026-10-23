"""Opt in to optimized full-KV W1 with raw-QK TopK by default.

Usage: PYTHONPATH=.../ra_harness python run_full_w1_optimized.py [eval_hf args]
RA_FULL_W1_SCORE_MODE=probability restores historical selector arithmetic.
RA_FULL_W1_RANK_BACKEND defaults to stable_sort; packed_topk requires probability.
"""
import os
import runpy
from pathlib import Path
import sys

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base / 'ra_harness'), str(base / 'ra_harness/kvcompress/harness')]
from kvcompress.engine.full_w1_optimized_cache import install

backend = os.environ.get('RA_FULL_W1_RANK_BACKEND', 'stable_sort')
score_mode = os.environ.get('RA_FULL_W1_SCORE_MODE', 'qk')
install(rank_backend=backend, score_mode=score_mode)
implementation = ('full_w1_fixed_ranges_batched_qk_v1' if score_mode == 'qk'
                  else 'full_w1_fixed_ranges_batched_exact_v1')
print(f'[full-W1 implementation] {implementation} '
      f'rank_backend={backend} score_mode={score_mode}', flush=True)
runpy.run_path(str(base / 'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
