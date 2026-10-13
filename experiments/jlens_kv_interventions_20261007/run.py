"""Full-context causal pilot. Execution is gated before model loading.

Interventions modify a single block-output residual row during prefill only.
Decode is greedy and Dense, with no answer-aware decode injection.
"""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import os
import sys
import threading
import time
from prepare import HERE, read, save, sha


def atomic_json(path, obj):
    p = Path(path)
    temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    save(temp, obj)
    temp.replace(p)


def frozen_source():
    source = (HERE / read(HERE / 'config.json')['source_experiment']).resolve()
    spec = importlib.util.spec_from_file_location('_kv_causal_source_common', source / 'common.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate():
    approval = read(HERE / 'approval.json')
    assert approval.get('approved') is True
    assert approval['lock_sha256'] == sha(HERE / 'lock.json')
    for path, expected in read(HERE / 'lock.json')['files'].items():
        assert sha(path) == expected, path
    assert os.environ.get('SLURM_JOB_ID'), 'Model execution needs Slurm'


def forward(model, ids, position_ids=None, mask=None):
    import torch
    from transformers import DynamicCache
    cache = DynamicCache()
    n = len(ids)
    positions = list(range(n)) if position_ids is None else position_ids
    mask = [1] * n if mask is None else mask
    result = model.model(input_ids=torch.tensor([ids], device='cuda'),
                         position_ids=torch.tensor([positions], device='cuda'),
                         attention_mask=torch.tensor([mask], device='cuda'),
                         past_key_values=cache, use_cache=True, return_dict=True)
    logits = model.lm_head(result.last_hidden_state[:, -1]).float()[0]
    assert torch.isfinite(logits).all()
    return logits, cache


def run(index):
    gate()  # No GPU operation occurs before approval is validated.
    import numpy as np
    import torch
    from intervention_math import change, token_atoms
    c = read(HERE / 'config.json')
    p = read(HERE / 'prepared.json')
    trial = p['plan'][index]
    if index >= 3:
        for baseline in p['plan'][:3]:
            folder = HERE / 'outputs' / ('trial-%03d' % baseline['index'])
            reference = read(folder / 'result.json')
            assert reference['status'] == 'complete'
            assert reference['historical_baseline_generated_prefix_match'] is True
            assert reference['lock_sha256'] == sha(HERE / 'lock.json')
            assert read(folder / 'analysis.json')['status'] == 'complete'
    directory = HERE / 'outputs' / f'trial-{index:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        result = read(directory / 'result.json')
        assert sha(directory / 'activations.pt') == result['artifact_sha256']
        return
    common = frozen_source()
    common.runtime()
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, path
    for path, expected in p['model_files'].items():
        stat = Path(path).stat()
        assert stat.st_size == expected['bytes'] and stat.st_mtime_ns == expected['mtime_ns'], path
    old = common.legacy()
    prior, historical_manifest = old.preflight()
    heads = old.load_heads(prior, historical_manifest)
    model, tokenizer = common.load_model()
    old.runtime_quality().install_mlp(model)
    parameter_versions = {name:v._version for name,v in model.named_parameters()}
    from adapter import install
    method = trial['method']
    spec = prior['scenarios'].get(method)
    state = install(model, spec['selector_method'] if spec else method,
                    layers=prior['layers'], heads=heads[spec['head_strategy']] if spec else None,
                    options=prior['baselines'] if method in prior['baselines'] else prior['selection'])
    ids = np.fromfile(p['inputs']['ids'], dtype='<i8').tolist()
    input_positions = np.fromfile(p['inputs']['pos'], dtype='<i8').tolist()
    mask = np.fromfile(p['inputs']['mask'], dtype='u1').tolist()
    modified = trial['kind'].startswith('text_')
    if modified:
        text_spec = p['text_inputs'][trial['site'] + '-' + trial['kind'].split('_')[1]]
        assert sha(text_spec['ids']) == text_spec['sha256']
        ids = np.fromfile(text_spec['ids'], dtype='<i8').tolist()
        input_positions, mask = list(range(len(ids))), [1] * len(ids)
    activation_trial = trial['kind'] not in ['baseline','text_correct','text_wrong']
    original_n = p['item']['input_tokens']
    positions = p['item']['analysis_positions'] if not modified else sorted(set(
        range(min(16,len(ids)))) | set(range(len(ids)-128,len(ids))) |
        set(int(x) for x in np.linspace(0,len(ids)-1,32)))
    indices = torch.tensor(positions, device='cuda')
    residuals, decode_residuals, intervention_record = {}, {}, {}
    phase = {'value':'prefill','step':-1}
    handles = []
    lens = torch.load(p['lens'], map_location='cpu', weights_only=True, mmap=True)
    donor, atoms, wrong_atoms = None, None, None
    generator = torch.Generator(device='cuda').manual_seed(c['seed'])
    if activation_trial:
        layer = trial['layer']
        unique = sorted(set(p['target_ids']))
        matrix = lens['J'][layer].float().cuda()
        atoms = token_atoms(model.lm_head.weight, model.model.norm.weight, matrix, unique)
        wrong_atoms = token_atoms(model.lm_head.weight, model.model.norm.weight, matrix, p['wrong_ids'])
        dense = torch.load(p['artifacts']['dense_full'], map_location='cpu', weights_only=True, mmap=True)
        position = p['sites'][trial['site']]
        donor = dense['residuals'][layer][dense['positions'].index(position)].float().cuda()
        del dense, matrix

    def hook_for(layer):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if phase['value'] == 'prefill':
                assert tensor.shape[1] == len(ids)
                if activation_trial and layer == trial['layer']:
                    pos = p['sites'][trial['site']]
                    before = tensor[0,pos].float().clone()
                    after, delta = change(before, kind=trial['kind'], alpha=trial['alpha'],
                                          atoms=atoms, ordered_ids=p['target_ids'], unique_ids=unique,
                                          wrong_atoms=wrong_atoms, donor=donor, generator=generator)
                    tensor[0,pos] = after.to(tensor.dtype)
                    intervention_record.update(layer=layer, position=pos, applied_rows=1,
                                               before=before.cpu(), after=tensor[0,pos].detach().cpu(),
                                               requested_delta=delta.cpu(),
                                               requested_relative_norm=float(delta.norm()/before.norm()))
                residuals[layer] = tensor[0].index_select(0,indices).detach().cpu()
            else:
                assert tensor.shape[1] == 1
                decode_residuals.setdefault(layer,[]).append(tensor[0,0].detach().cpu())
            return output
        return hook

    for layer in range(32):
        handles.append(model.model.layers[layer].register_forward_hook(hook_for(layer)))
    progress = {'trial':trial,'stage':'prefill','completed_decode_steps':0,'input_tokens':len(ids)}
    stop = threading.Event()
    def snapshots():
        while not stop.wait(c['snapshot_seconds']):
            atomic_json(directory / 'progress.json', dict(progress))
    thread = threading.Thread(target=snapshots, daemon=True)
    thread.start()
    atomic_json(directory / 'progress.json', progress)
    started = time.monotonic()
    try:
        with torch.no_grad():
            logits, cache = forward(model, ids, input_positions, mask)
            initial_logits = logits.detach().cpu()
            phase['value'] = 'decode'
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos,int) else list(eos)
            generated = []
            for step in range(c['generation']['max_new_tokens']):
                token = int(logits.argmax())
                generated.append(token)
                progress.update(stage='decode',completed_decode_steps=len(generated))
                if token in eos or step == c['generation']['max_new_tokens']-1:
                    break
                offset = len(ids)+step
                phase['step'] = step
                out = model.model(input_ids=torch.tensor([[token]],device='cuda'),
                                  position_ids=torch.tensor([[offset]],device='cuda'),
                                  attention_mask=torch.ones(1,offset+1,dtype=torch.long,device='cuda'),
                                  past_key_values=cache,use_cache=True,return_dict=True)
                logits = model.lm_head(out.last_hidden_state[:,-1]).float()[0]
                assert torch.isfinite(logits).all()
                assert all(k.shape == v.shape == (1,8,offset+1,128)
                           for k,v in zip(cache.key_cache,cache.value_cache))
                if step % 8 == 0:
                    atomic_json(directory / 'generation-checkpoint.json',dict(
                        generated_ids=generated, prediction=tokenizer.decode(generated,skip_special_tokens=True),
                        resumable_kv_cache=False, purpose='partial-result preservation; failed trials rerun from prefill'))
            torch.cuda.synchronize()
        assert set(residuals) == set(range(32))
        assert len(intervention_record) > 0 if activation_trial else not intervention_record
        assert set(state['prefill']) == set(range(32))
        assert all(state['decode_calls'].get(l,0) == len(generated)-1 for l in range(32))
        assert parameter_versions == {name:v._version for name,v in model.named_parameters()}
        scorer = old.official(old.ORIGIN / 'official')[2][p['item']['task']]
        prediction = tokenizer.decode(generated,skip_special_tokens=True)
        short_prediction = tokenizer.decode(generated[:c['generation']['also_score_prefix_tokens']],skip_special_tokens=True)
        artifact = dict(prefill_positions=positions,prefill_residuals=residuals,
                        decode_positions=list(range(len(ids),len(ids)+len(generated)-1)),
                        decode_residuals={l:torch.stack(hs) for l,hs in decode_residuals.items()},
                        intervention=intervention_record, initial_logits=initial_logits,
                        generated_ids=generated)
        common.atomic_torch_save(directory / 'activations.pt',artifact)
        history_match = None
        if trial['kind'] == 'baseline':
            hist = read(p['item']['history'][method]['path'])
            original_steps = len(hist['generated_ids'])
            history_match = generated[:original_steps] == hist['generated_ids']
            if not history_match:
                raise RuntimeError('Fresh baseline does not reproduce the historical generated prefix; review before launching interventions')
        atomic_json(directory / 'result.json',dict(
            status='complete',trial=trial, key=p['item']['key'],input_tokens=len(ids),
            original_input_tokens=original_n, oracle_intervention=trial['kind']!='baseline',
            prediction=prediction, generated_ids=generated,
            score=float(scorer(prediction,p['item']['answers'],'llama')),
            score_first_50_tokens=float(scorer(short_prediction,p['item']['answers'],'llama')),
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            historical_baseline_generated_prefix_match=history_match,
            inject_during_decode=False,dense_decode=True,
            block_output_intervention_changes_this_and_later_residuals=True,
            current_block_kv_precedes_output_hook=True,
            prefill_stats={str(k):v for k,v in state['prefill'].items()},
            intervention_relative_norm=intervention_record.get('requested_relative_norm'),
            seconds=time.monotonic()-started,artifact_sha256=sha(directory/'activations.pt'),
            approval_sha256=sha(HERE/'approval.json'),lock_sha256=sha(HERE/'lock.json'),
            job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0)))
        print('COMPLETE',index,trial,prediction,flush=True)
    except Exception as exc:
        atomic_json(directory/'failure.json',dict(error=repr(exc),trial=trial))
        raise
    finally:
        stop.set();thread.join()
        atomic_json(directory/'progress.json',dict(progress))
        state['close']()
        for h in handles:
            h.remove()
        del model,lens
        gc.collect();torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',type=int,required=True)
    run(parser.parse_args().index)
