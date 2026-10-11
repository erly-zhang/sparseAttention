"""Opt in to current-query raw-QK TopK; use a separate output directory.

Accepts the ordinary eval_hf arguments, with grid_full_w1_topk_protected or
grid_full_w1_topk_none. Probability-mode runners remain unchanged. Only the
selector skips scaling/softmax; attention still normalizes selected QK scores.
"""
import runpy
from pathlib import Path
import sys

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base / 'ra_harness'), str(base / 'ra_harness/kvcompress/harness')]
from kvcompress.engine.full_w1_optimized_cache import install

install(rank_backend='stable_sort', score_mode='qk')
print('[full-W1 implementation] full_w1_fixed_ranges_batched_qk_v1 '
      'rank_backend=stable_sort score_mode=qk', flush=True)
runpy.run_path(str(base / 'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
