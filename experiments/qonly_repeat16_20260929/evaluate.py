"""Frozen KV50 diagnostic: dense, Q-only/native16, Q-only/repeat16."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from config import METHODS, ROOT, install, protocol, validate_counts

HERE = Path(__file__).resolve().parent
PRIOR = ROOT / 'experiments/psc_kl_quality_retry02_20260927'
ABLATION = ROOT / 'experiments/head_slot_kv_ablation_20260928'
SMOKE_IDS = (331, 61, 16)
PINS = {
    PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
    PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
    PRIOR / 'inputs/manifest.json': 'dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3',
    ABLATION / 'inputs/manifest.json': 'abdf9a76ce3565cb1ce65c1231b90ec9a1857c0816a2a7146057ddcf69151dbc',
    ABLATION / 'legacy_checks.py': '50011b36274403486429a27bc8f90d48e910e804d034c8a92fae0f55a716b297',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def preflight():
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE / name) == digest, name
    q = load('repeat16_quality', PRIOR / 'quality.py')
    q.verify_sources()
    old = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, digest in old['tokenizer_files'].items():
        assert sha(q.MODEL / name) == digest
    score, digest = q.scorer()
    assert digest == old['scorer_sha256']
    assert sha(q.BASE / 'data/kv_retrieval.jsonl') == old['dataset_sha256']
    return q, old, score


def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    q, old, _ = preflight()
    previous = json.loads((ABLATION / 'inputs/manifest.json').read_text())
    records = [r for r in previous['records'] if r['task'] == 'kv_retrieval']
    assert len(records) == len({r['source_index'] for r in records}) == 50
    checks = load('repeat16_checks', ABLATION / 'legacy_checks.py')
    assert [r['source_index'] for r in records] == checks.select_indices(old['indices'])
    data = [json.loads(line) for line in (q.BASE / 'data/kv_retrieval.jsonl').read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    (HERE / 'inputs').mkdir()
    for row in records:
        index = row['source_index']
        path = ABLATION / 'inputs' / f'kv_retrieval-{index}.ids'
        raw = path.read_bytes()
        assert sha(path) == row['input_ids_sha256'] and len(raw) == 8 * row['input_tokens']
        ids, before = q.render_ids(tok, data[index])
        assert ids == np.frombuffer(raw, dtype='<i8').tolist()
        assert before == row['before_truncation'] and data[index]['answer'] == row['answers']
        with (HERE / 'inputs' / f'{index}.ids').open('xb') as handle:
            handle.write(raw)
    save(HERE / 'inputs/manifest.json', dict(records=records, source_manifest_sha256=sha(HERE/'SHA256SUMS'),
        prior_manifest_sha256=sha(ABLATION/'inputs/manifest.json'), tokenizer_files=old['tokenizer_files'],
        scorer_sha256=old['scorer_sha256'], dataset_sha256=old['dataset_sha256'],
        model=str(q.MODEL), generation='greedy seed42 max128',
        limitation='Reused50 diagnostic inputs; no training or tuning; not an unseen benchmark'))
    print('FROZEN KV50', sha(HERE/'inputs/manifest.json'), flush=True)


def setup():
    q, _, score = preflight()
    m = json.loads((HERE/'inputs/manifest.json').read_text())
    assert m['source_manifest_sha256'] == sha(HERE/'SHA256SUMS')
    assert len(m['records']) == len({r['input_ids_sha256'] for r in m['records']}) == 50
    for row in m['records']:
        p = HERE/'inputs'/f"{row['source_index']}.ids"
        assert sha(p) == row['input_ids_sha256'] and p.stat().st_size == 8*row['input_tokens']
    return q, m, score


def selected(manifest, phase):
    if phase not in ('smoke', 'remaining', None):
        raise ValueError(phase)
    return [r for r in manifest['records'] if phase is None or
            (r['source_index'] in SMOKE_IDS) == (phase == 'smoke')]


def validate_rows(rows, inputs):
    expected = {r['source_index']: r for r in inputs}
    assert len(expected) == len(inputs) and len(rows) == len(inputs)*len(METHODS)
    seen = set()
    for row in rows:
        key = row['method'], row['source_index']
        assert key not in seen and key[0] in METHODS and key[1] in expected
        seen.add(key)
        item = expected[key[1]]
        for field in ('input_ids_sha256','input_tokens','answers','raw_row_sha256'):
            assert row[field] == item[field]
        assert row['full_kv_cache'] and row['finite_all_steps'] and row['dense_decode']
        assert 1 <= len(row['generated_ids']) == row['generated_tokens'] <= 128
        validate_counts(key[0], row['routed_layer_eligible_pairs'], row['input_tokens'])
        if key[0] == 'dense' and key[1] in SMOKE_IDS:
            assert row['historical_token_ids_equal'] is True
    assert seen == {(method,index) for method in METHODS for index in expected}


def worker(method, phase):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
    assert transformers.__version__.startswith('4.44.')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    q, manifest, score = setup()
    checks = load('repeat16_diagnostics', ABLATION/'legacy_checks.py')
    model = AutoModelForCausalLM.from_pretrained(q.MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    q.install_mlp(model)  # Frozen FFN memory implementation, not a learned adapter.
    state = install(model,method)
    tok = AutoTokenizer.from_pretrained(q.MODEL,local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else list(eos)

    def step(tokens, cache, offset):
        with torch.inference_mode():
            output = model.model(input_ids=tokens,position_ids=torch.arange(offset,offset+tokens.shape[1],device='cuda')[None],
                attention_mask=torch.ones(1,offset+tokens.shape[1],dtype=torch.long,device='cuda'),
                past_key_values=cache,use_cache=True,return_dict=True)
            logits = model.lm_head(output.last_hidden_state[:,-1:]).float()[0,0]
        assert bool(torch.isfinite(logits).all())
        assert len(cache.key_cache) == len(cache.value_cache) == 32
        assert all(k.shape == v.shape == (1,8,offset+tokens.shape[1],128)
                   for k,v in zip(cache.key_cache,cache.value_cache))
        return logits

    cache = DynamicCache()
    step(torch.tensor([[1,2,3,4]],device='cuda'),cache,0)
    del cache
    for item in selected(manifest,phase):
        name = f"{method}-{item['source_index']}"
        ids = torch.tensor(np.fromfile(HERE/'inputs'/f"{item['source_index']}.ids",dtype='<i8'),device='cuda')[None]
        n = ids.shape[1]
        cache = DynamicCache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        logits = step(ids,cache,0)
        torch.cuda.synchronize()
        prefill = time.monotonic()-start
        path = HERE/'results'/f'{name}.npy'
        with path.open('xb') as handle:
            np.save(handle,logits.detach().cpu().double().numpy(),allow_pickle=False)
        torch.cuda.synchronize()
        start = time.monotonic()
        generated = []
        for j in range(128):
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j == 127:
                break
            logits = step(torch.tensor([[token]],device='cuda'),cache,n+j)
        torch.cuda.synchronize()
        decode = time.monotonic()-start
        prediction = tok.decode(generated,skip_special_tokens=True)
        historical = None
        if method == 'dense' and item['source_index'] in SMOKE_IDS:
            old = json.loads((PRIOR/f"formal/{item['source_index']}/dense/result.json").read_text())
            assert old['input_ids_sha256'] == item['input_ids_sha256']
            historical = old['generated_ids'] == generated
        counts = {} if state is None else dict(state['pairs'])
        validate_counts(method,counts,n)
        row = dict(**item,method=method,phase=phase,generated_ids=generated,generated_tokens=len(generated),
            prediction=prediction,correct=int(score(('===answerspliter==='.join(item['answers']),prediction))),
            eos_ids=eos,stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            prefill_seconds=prefill,decode_seconds=decode,repetition=checks.repetition(generated),
            full_kv_cache=True,dense_decode=True,finite_all_steps=True,routed_layer_eligible_pairs=counts,
            historical_token_ids_equal=historical,logits_sha256=sha(path),
            gpu=torch.cuda.get_device_name(),node=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(HERE/'results'/f'{name}.json',row)
        print('RESULT',method,item['source_index'],row['correct'],len(generated),round(prefill+decode,3),flush=True)
        assert historical is not False, 'Dense historical generated IDs changed'
        del ids,cache,logits
        gc.collect()
        torch.cuda.empty_cache()
    if state:
        state['close']()


def audit(phase=None):
    import numpy as np
    _, manifest, score = setup()
    checks = load('repeat16_score_checks', ABLATION/'legacy_checks.py')
    inputs = selected(manifest,phase)
    rows = [json.loads((HERE/'results'/f"{method}-{item['source_index']}.json").read_text())
            for item in inputs for method in METHODS]
    validate_rows(rows,inputs)
    for row in rows:
        assert row['correct'] == score(('===answerspliter==='.join(row['answers']),row['prediction']))
        assert row['repetition'] == checks.repetition(row['generated_ids'])
        path = HERE/'results'/f"{row['method']}-{row['source_index']}.npy"
        assert sha(path) == row['logits_sha256']
        a = np.load(HERE/'results'/f"dense-{row['source_index']}.npy",allow_pickle=False)
        b = np.load(path,allow_pickle=False)
        assert a.shape == b.shape == (128256,) and np.isfinite(a).all() and np.isfinite(b).all()
        def logp(x):
            x = x.astype(np.float64)-x.max()
            return x-np.log(np.exp(x).sum())
        a,b = logp(a),logp(b)
        kl = float((np.exp(a)*(a-b)).sum())
        assert kl >= -1e-10
        row['final_prompt_vocabulary_kl'] = kl
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        summary[method] = dict(count=len(group),correct=sum(r['correct'] for r in group),
            capped=sum(r['stop_reason']=='max_new_tokens' for r in group),
            repetitive=sum(r['repetition']['repetitive_flag'] for r in group),
            mean_prefill_seconds=statistics.mean(r['prefill_seconds'] for r in group),
            mean_final_prompt_vocabulary_kl=statistics.mean(r['final_prompt_vocabulary_kl'] for r in group))
    return dict(records=rows,summary=summary,manifest_sha256=sha(HERE/'inputs/manifest.json'))


def projection(elapsed, rows, remaining=47):
    return elapsed + 1.5*remaining*sum(max(r['prefill_seconds']+r['decode_seconds']
        for r in rows if r['method']==m) for m in METHODS) + 1200


def run():
    _, manifest, _ = setup()
    assert os.environ.get('SLURM_JOB_ID')
    (HERE/'results').mkdir()
    save(HERE/'protocol.json',dict(**protocol(),manifest_sha256=sha(HERE/'inputs/manifest.json'),
        resource='1 H10080GB,8CPU,128GB,max2h',seed=42,max_new_tokens=128))
    start = time.monotonic()
    def phase_run(phase):
        for method in METHODS:
            budget = 6600-(time.monotonic()-start)
            assert budget > 0
            subprocess.run([sys.executable,__file__,'--method',method,'--phase',phase],check=True,timeout=budget)
    phase_run('smoke')
    report = audit('smoke')
    projected = projection(time.monotonic()-start,report['records'])
    assert projected < 6600, f'Budget gate rejected: {projected}'
    save(HERE/'smoke-complete.json',dict(**report,status='passed',projected_total_seconds=projected))
    print('SMOKE PASS9; PROJECTION',projected,flush=True)
    phase_run('remaining')
    report = audit()
    assert len(list((HERE/'results').glob('*.json'))) == len(report['records']) == 150
    save(HERE/'complete.json',dict(**report,status='complete',elapsed_seconds=time.monotonic()-start))
    print('REPEAT16 COMPLETE150',json.dumps(report['summary']),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--phase',choices=('smoke','remaining'))
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare()
        elif args.check:
            setup()
            print('FROZEN PREFLIGHT PASS',flush=True)
        elif args.audit:
            print(json.dumps(audit()['summary']))
        elif args.method:
            assert args.phase
            worker(args.method,args.phase)
        else:
            run()
    except BaseException as error:
        if not (HERE/'failure.json').exists():
            save(HERE/'failure.json',dict(error=repr(error),job=os.environ.get('SLURM_JOB_ID'),args=vars(args)))
        raise
