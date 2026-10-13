"""Isolated PSC interleaved prefill diagnostic; no generation or training."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path('/ocean/projects/cis260309p/jzhang114/sparseAttention')
BASE = ROOT / 'experiments/psc_kv_layer_outputs_20260925'
OLD = ROOT / 'experiments/psc_kv_smoke_version_kl_20260925'
OUT = ROOT / 'experiments/psc_interleaved_kl_20260926'
MODEL = ROOT.parent / 'prefill-routing-deps/models/Llama-3.1-8B-Instruct'
DENSE = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31]
ROUTING = [i for i in range(32) if i not in DENSE]
IDENTITY = 'a7ec726b2a8a55b947a3a7224f4d952862fdb8d7e74c682f74da441dc360a9c9'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache, LlamaConfig, LlamaForCausalLM

    assert os.environ.get('SLURM_JOB_ID') == '47149101'
    assert torch.cuda.device_count() == 1 and 'L40S' in torch.cuda.get_device_name()
    assert len(DENSE) == 11 and len(ROUTING) == 21
    assert not OUT.exists(), 'Refuse duplicate execution or overwrite'
    OUT.mkdir()
    start = time.monotonic()
    save('state.json', {'status': 'checking', 'job': os.environ['SLURM_JOB_ID']})
    shutil.copyfile(__file__, OUT / 'source.py')
    try:
        source = ROOT / 'experiments/psc_kv_hybrid_smoke_20260925/vscode_l40s/smoke.py'
        frozen = json.loads((OLD / 'source_sha256.json').read_text())
        for name, expected in frozen.items():
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            assert sha(path) == expected, ('Frozen source changed', name)
        spec = importlib.util.spec_from_file_location('verified_smoke', source)
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        base = smoke.install.__globals__['base']
        sys.path.insert(0, str(ROOT / 'experiments/prefill_token_head_routing_20260924/layer_output_comparison'))
        from metrics import compare_outputs

        def install(model):
            original = [layer.self_attn.forward for layer in model.model.layers]
            model, state = base.install(model, 'routing_c')
            for i in DENSE:
                model.model.layers[i].self_attn.forward = original[i]
            assert all((layer.self_attn.forward == original[i]) == (i in DENSE)
                       for i, layer in enumerate(model.model.layers))
            return state

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4)
        cfg = LlamaConfig(hidden_size=512, intermediate_size=768, num_hidden_layers=32,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                          max_position_embeddings=512, torch_dtype=torch.bfloat16)
        cfg._attn_implementation = 'flash_attention_2'
        tiny = LlamaForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
        smoke.install_memory_mlp(tiny)
        state = install(tiny)
        with torch.inference_mode():
            probe = tiny.model(input_ids=torch.arange(33, device='cuda')[None], use_cache=True)
        assert torch.isfinite(probe.last_hidden_state).all()
        assert len(probe.past_key_values) == 32
        assert all(k.shape == (1, 2, 33, 128) for k, v in probe.past_key_values)
        assert all(int(state['pairs'][i]) > 0 for i in ROUTING)
        save('correctness.json', {'status': 'passed', 'dense_layers': DENSE,
                                  'routing_layers': ROUTING, 'full_tiny_cache': True})
        del tiny, probe, state
        torch.cuda.empty_cache()
        provenance = json.loads((BASE / 'model_provenance.json').read_text())
        model_hashes = {name: sha(MODEL / name) for name in provenance['files']}
        assert all(model_hashes[n] == v['sha256'] for n, v in provenance['files'].items())
        raw = (BASE / 'inputs/3.ids.i64le').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == IDENTITY
        ids = torch.tensor(np.frombuffer(raw, dtype='<i8').copy(), device='cuda')[None]
        assert ids.shape == (1, 123959)
        save('provenance.json', {'input_ids_sha256': IDENTITY, 'input_index': 3,
             'tokens': 123959, 'model_files': model_hashes, 'revision': provenance['revision'],
             'source_sha256': sha(Path(__file__)), 'frozen_dependencies': frozen,
             'gpu': torch.cuda.get_device_name(), 'node': os.uname().nodename,
             'dense_layers': DENSE, 'routing_layers': ROUTING,
             'KL': 'mean token KL(softmax(dense)||softmax(method)), T=1, nats',
             'exploratory_posthoc': True, 'own_end_to_end': True, 'no_generation': True,
             'mlp_buffer_reuse_both_conditions': True, 'not_inference_timing': True})
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                  torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
        smoke.install_memory_mlp(model)
        references = OUT / 'dense_tensors'
        references.mkdir()
        all_rows = []
        for method in ['dense', 'interleaved']:
            if method == 'interleaved':
                state = install(model)
            with torch.inference_mode():
                warm = model.model(input_ids=ids[:, :129], use_cache=True)
            del warm
            rows = []
            handles = []
            def capture(layer, point):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    cpu = value.detach().cpu()
                    assert list(cpu.shape) == [1, 123959, 4096]
                    path = references / f'{point}_{layer:02d}.pt'
                    if method == 'dense':
                        torch.save(cpu, path)
                        reference = cpu
                    else:
                        reference = torch.load(path, map_location='cpu', weights_only=True)
                    metrics = compare_outputs(reference, cpu, torch.ones(cpu.shape[:2], dtype=torch.bool))
                    assert math.isfinite(metrics['kl_dense_to_method'])
                    row = dict(method=method, layer=layer, point=point,
                               input_ids_sha256=IDENTITY, **metrics)
                    rows.append(row)
                    with (OUT / 'partial.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                return hook
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.o_proj.register_forward_hook(capture(i, 'attention_post_o_proj')))
                handles.append(layer.register_forward_hook(capture(i, 'block_output')))
            with torch.inference_mode():
                cache = DynamicCache()
                result = model.model(input_ids=ids, position_ids=torch.arange(123959, device='cuda')[None],
                    attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            assert torch.isfinite(result.last_hidden_state).all()
            assert len(cache.key_cache) == 32
            assert all(list(k.shape) == [1, 8, 123959, 128] for k in cache.key_cache)
            for handle in handles:
                handle.remove()
            assert len(rows) == 64
            for row in rows:
                i = row['layer']
                row['retained_causal_pairs'] = (32 * 123959 * 123960 // 2 if method == 'dense' or i in DENSE
                                                else int(state['pairs'][i]))
            all_rows.extend(rows)
            del result, cache
            print(method, 'finished', flush=True)
        assert len({(r['method'], r['layer'], r['point']) for r in all_rows}) == 128
        save('all_layers.json', all_rows)
        save('state.json', {'status': 'complete', 'records': 128, 'seconds': time.monotonic()-start,
                           'job': os.environ['SLURM_JOB_ID'], 'step': os.environ.get('SLURM_STEP_ID')})
    except BaseException:
        save('state.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise

if __name__ == '__main__':
    main()
