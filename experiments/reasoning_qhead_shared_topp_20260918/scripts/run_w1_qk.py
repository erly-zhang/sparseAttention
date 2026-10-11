"""Opt-in raw-QK W1 TopK for optimized full KV and compact500 shared KV."""
import runpy
import sys
from pathlib import Path

base=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(base/'ra_harness'),str(base/'ra_harness/kvcompress/harness')]
import kvcompress.engine.selector_grid_cache as grid
from kvcompress.engine.full_w1_optimized_cache import install

install(rank_backend='stable_sort',score_mode='qk')
original=grid.SelectorGridLayer
def factory(*args,**kwargs):
    mode=args[3] if len(args)>3 else kwargs.get('mode',kwargs.get('eviction_mode',''))
    if mode in ('grid_compact500_w1_topk_protected','grid_compact500_w1_topk_none'):
        kwargs['score_mode']='qk'
    return original(*args,**kwargs)
grid.SelectorGridLayer=factory
print('[W1 selector] raw QK, stable chronological ties; attention softmax retained',flush=True)
runpy.run_path(str(base/'ra_harness/kvcompress/harness/eval_hf.py'),run_name='__main__')
