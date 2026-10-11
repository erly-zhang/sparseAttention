"""Usual runner with process-local opt-in and post-timing token-ID capture."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

base = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(base/'ra_harness'), str(base/'ra_harness/kvcompress/harness')]
implementation = os.environ['RA_FULL_W1_TRIAL_IMPL']
backend = os.environ.get('RA_FULL_W1_RANK_BACKEND', 'stable_sort')
if implementation == 'optimized':
    from kvcompress.engine.full_w1_optimized_cache import install
    install(backend)
elif implementation != 'baseline':
    raise ValueError(implementation)
print(f'[full-W1 trial] implementation={implementation} rank_backend={backend}', flush=True)

# eval_hf imports this module by its top-level script name.
import generation_utils as generation
original = generation.batch_exist_generate
captures = []


def capture(*args, **kwargs):
    output = original(*args, **kwargs)
    metrics = kwargs.get('metrics')
    if metrics is not None:
        # Retain only the small output tensor. No copy, synchronization or file
        # write enters synchronized decode; transfer/serialization is at exit.
        captures.append((output, metrics))
        metrics['full_w1_implementation'] = implementation
        metrics['full_w1_rank_backend'] = backend if implementation == 'optimized' else 'historical_stable_sort'
    return output


generation.batch_exist_generate = capture


def save_tokens():
    target = Path(os.environ['RA_FULL_W1_TOKEN_OUTPUT'])
    rows = []
    for output, metrics in captures:
        arrays = output.detach().cpu().numpy().astype('<i8')
        rows.append(dict(idx=metrics['idx'], input_tokens=metrics['input_tokens'],
                         token_ids=arrays.tolist(),
                         token_sha256_per_row=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]))
    with target.open('x') as f:
        json.dump(rows, f)
        f.write('\n')


atexit.register(save_tokens)
runpy.run_path(str(base/'ra_harness/kvcompress/harness/eval_hf.py'), run_name='__main__')
