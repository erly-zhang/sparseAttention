"""Isolated exact raw-QK W1 implementation optimizations (full/hybrid1000)."""
import os
import atexit
import json
import runpy
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base/'ra_harness'), str(base/'ra_harness/kvcompress/harness')]
from kvcompress.engine.w1_qk_fast_cache import install
backend = os.environ.get('RA_W1_QK_FAST_BACKEND', 'packed_qk_topk')
install(backend)
print(f'[W1 raw-QK fast v2] backend={backend}; exact stable ties; historical RNG', flush=True)
if os.environ.get('RA_W1_TOKEN_OUTPUT'):
    import generation_utils as generation
    original = generation.batch_exist_generate
    captures = []

    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs.get('metrics') is not None:
            captures.append((output, kwargs['metrics']))
        return output

    generation.batch_exist_generate = capture

    def save_tokens():
        records = [dict(idx=m['idx'], token_ids=out.detach().cpu().tolist()) for out, m in captures]
        with Path(os.environ['RA_W1_TOKEN_OUTPUT']).open('x') as target:
            json.dump(records, target)
            target.write('\n')

    atexit.register(save_tokens)
runpy.run_path(str(base/'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
