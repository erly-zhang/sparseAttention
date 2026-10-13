"""Pair ordinary and observed prefill in one process on the first failed input."""
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'representative_patterns_fast_retry03_20261001'
sys.path.insert(0, str(ORIGINAL))
from protocol import sha, save
from run import preflight, runtime_quality, forward_prompt
from adapter import install
from observer import HeadObserver


def run():
    c, m = preflight()
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, '/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')
    flex = importlib.import_module('flex_prefill.ops.flex_prefill_attention')
    baselines = importlib.import_module('baselines')
    model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    runtime_quality().install_mlp(model)
    item = next(x for x in m['evaluation'] if x['key'] == 'code_debug-48')
    previous = ORIGINAL / 'outputs/formal-5/code_debug-48-flexprefill.json'
    old = json.loads(previous.read_text())
    historical = np.load(previous.parent / old['logits_file'], allow_pickle=False)
    out = HERE / 'diagnosis'
    out.mkdir(exist_ok=False)
    results = {}
    arrays = {}
    for mode in ['plain_first', 'observed', 'plain_repeat']:
        state = install(model, 'flexprefill', layers=c['baseline_layers'], options=c['baselines'])
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        start = time.monotonic()
        try:
            if mode == 'observed':
                with HeadObserver(flex, baselines) as observation:
                    logits, cache, _ = forward_prompt(model, item, state, lambda: None)
                layers = {str(k): v for k, v in observation.records.items()}
            else:
                logits, cache, _ = forward_prompt(model, item, state, lambda: None)
                layers = None
            array = logits.cpu().numpy().copy()
            arrays[mode] = array
            results[mode] = {'historical_exact': bool(np.array_equal(array, historical)),
                'historical_max_abs_error': float(np.abs(array - historical).max()),
                'elapsed_seconds': time.monotonic() - start, 'layers': layers,
                'argmax': int(array.argmax()), 'historical_argmax': int(historical.argmax())}
            np.save(out / (mode + '.npy'), array, allow_pickle=False)
            save(out / (mode + '.json'), results[mode])
            print('DIAGNOSIS', mode, {k: v for k, v in results[mode].items() if k != 'layers'}, flush=True)
        finally:
            state['close']()
        del logits, cache
        gc.collect()
        torch.cuda.empty_cache()
    comparison = {}
    for first, second in [('plain_first', 'observed'), ('plain_first', 'plain_repeat'), ('observed', 'plain_repeat')]:
        comparison[first + '_vs_' + second] = {'bitwise_equal': bool(np.array_equal(arrays[first], arrays[second])),
            'max_abs_error': float(np.abs(arrays[first] - arrays[second]).max())}
    save(out / 'comparison.json', {'comparisons': comparison, 'passes': results})
    print('COMPARISON', comparison, flush=True)


if __name__ == '__main__':
    run()
