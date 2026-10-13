"""Official sparse kernels through the common RoPE/full-cache/dense-decode adapter.

Imports that allocate CUDA state are deferred until approved GPU execution.
"""
import json
from functools import lru_cache
import sys
from pathlib import Path
import torch

FLEX = Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')


@lru_cache(None)
def pattern(path):
    return json.loads(Path(path).read_text())


def baseline_attention(q, k, v, method, layer, options):
    if method == 'flexprefill':
        if str(FLEX) not in sys.path:
            sys.path.insert(0, str(FLEX))
        from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention
        output = flex_prefill_attention(q,k,v,**options[method])
    elif method == 'minference':
        from minference.modules.minference_forward import minference_prefill_kernel
        best = pattern(options[method]['pattern_file'])
        output = torch.empty_like(q)
        h, hk = q.shape[2], k.shape[2]
        for head in range(h):
            kv = head//(h//hk)
            value = minference_prefill_kernel(q[:,:,head:head+1].transpose(1,2),
                k[:,:,kv:kv+1].transpose(1,2), v[:,:,kv:kv+1].transpose(1,2), head, layer,
                {'best_pattern':best,'minference_ratio':options[method]['ratio']})
            output[:,:,head:head+1] = value.transpose(1,2)
    else:
        raise ValueError(method)
    if output.shape != q.shape or not bool(torch.isfinite(output).all()):
        raise ValueError('invalid baseline output')
    return output, dict(mode=method, implementation='official kernel with common Llama adapter',
        baseline_configuration=options[method], own_qk_renormalized=True, probabilities_shared=False,
        full_kv_cache=True, mask_density=None)
