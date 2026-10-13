import json
import os
from pathlib import Path

import torch
import minference.cuda
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
from adapters import install

assert torch.cuda.device_count() == 1
torch.set_num_threads(4)
results = {}
for method in ('flexprefill', 'minference'):
    torch.manual_seed(42)
    cfg = LlamaConfig(hidden_size=512, intermediate_size=768, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                      max_position_embeddings=512)
    cfg._attn_implementation = 'flash_attention_2'
    model = LlamaForCausalLM(cfg).to(device='cuda', dtype=torch.bfloat16).eval()
    ids = torch.arange(129, device='cuda').remainder(63)[None]

    def forward(tokens):
        cache = DynamicCache()
        with torch.inference_mode():
            out = model.model(tokens, position_ids=torch.arange(tokens.shape[1], device='cuda')[None],
                              attention_mask=torch.ones_like(tokens), past_key_values=cache,
                              use_cache=True).last_hidden_state
        assert len(cache.key_cache) == 2
        assert all(tuple(k.shape) == (1, 2, tokens.shape[1], 128) for k in cache.key_cache)
        return out

    reference = forward(ids)
    model, _ = install(model, method)
    actual = forward(ids)
    assert torch.isfinite(actual).all()
    relative = float((actual.float() - reference.float()).norm() / reference.float().norm())
    assert relative < .02, (method, relative)
    changed = ids.clone()
    changed[:, 65:] = (changed[:, 65:] + 7) % 63
    torch.testing.assert_close(actual[:, :65], forward(changed)[:, :65], rtol=.01, atol=.02)
    results[method] = dict(status='passed', relative_l2=relative,
                           full_prompt_cache=True, future_invariance=True)
    del model

loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if 'libstdc++' in line})
assert len(loaded) == 1 and 'v11.3.0' in loaded[0], loaded
result = dict(status='passed', fixture_only=True, gpu=torch.cuda.get_device_name(),
              cpp_runtime=loaded, ld_preload=os.environ['LD_PRELOAD'], methods=results,
              extension_path=minference.cuda.__file__)
assert Path(minference.cuda.__file__).is_relative_to(Path(__file__).parent / 'overlay')
with Path(__file__).with_name('probe_result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result), flush=True)
