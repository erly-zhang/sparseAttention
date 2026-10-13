"""Frozen six-generation shared-attention-map correctness pilot; no training."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PRIOR = EXPERIMENTS / 'psc_kl_quality_retry02_20260927'
OLD = EXPERIMENTS / 'qonly_mlp_calibration_20260928'
METHODS = ('dense_full', 'shared_map')
PINS = {
    PRIOR / 'quality.py': '7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61',
    PRIOR / 'env.sh': 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9',
    PRIOR / 'inputs/manifest.json': 'dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3',
    OLD / 'protocol.py': '26ac2c894bf63736f309551fd2a9e4832a81e44d938fafc49669c7da21f8be7f',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def specs():
    return [dict(id=f'smoke-{i:02d}', split='smoke', index=i,
                 seed=202609292000+i, length_bin=4096, depth=depth)
            for i, depth in enumerate((0.1, 0.5, 0.9))]


def preflight():
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        assert sha(HERE / name) == digest, name
    quality = load('shared_map_frozen_quality', PRIOR / 'quality.py')
    quality.verify_sources()
    prior = json.loads((PRIOR / 'inputs/manifest.json').read_text())
    for name, digest in prior['tokenizer_files'].items():
        assert sha(quality.MODEL / name) == digest
    return quality, prior


def encode(tok, prompt):
    return tok.encode(tok.apply_chat_template([dict(role='user', content=prompt)],
        tokenize=False, add_generation_prompt=True), add_special_tokens=False)


def prepare():
    from transformers import AutoTokenizer
    quality, prior = preflight()
    tok = AutoTokenizer.from_pretrained(quality.MODEL, local_files_only=True)
    generator = load('shared_map_synthetic_generator', OLD / 'protocol.py')
    (HERE / 'inputs').mkdir()
    rows = []
    for spec in specs():
        row = generator.make_prompt(spec, lambda prompt: encode(tok, prompt))
        ids = row.pop('ids')
        path = HERE / 'inputs' / (row['id'] + '.ids')
        with path.open('xb') as f:
            f.write(struct.pack('<'+'q'*len(ids), *ids))
        row.update(input_tokens=len(ids), input_ids_sha256=sha(path))
        rows.append(row)
    save(HERE / 'inputs/manifest.json', dict(records=rows,
        source_manifest_sha256=sha(HERE / 'SHA256SUMS'), model=str(quality.MODEL),
        tokenizer_files=prior['tokenizer_files'], methods=METHODS, layers=list(range(0,32,2)),
        algorithm='Full attention each head; position%32 row stitching; shared A times own GQA V',
        prefill_only=True, full_kv=True, dense_decode=True, training=False,
        seed=42, max_new_tokens=128, attention_arithmetic='FP32 in modified layers, BF16 output',
        budget=dict(gpus=1, gpu='H10080GB', cpus=8, memory_gb=128, max_seconds=1800)))
    check()
    print('FROZEN3', sha(HERE / 'inputs/manifest.json'), flush=True)


def check():
    from transformers import AutoTokenizer
    quality, prior = preflight()
    manifest = json.loads((HERE / 'inputs/manifest.json').read_text())
    assert manifest['source_manifest_sha256'] == sha(HERE / 'SHA256SUMS')
    assert manifest['tokenizer_files'] == prior['tokenizer_files']
    tok = AutoTokenizer.from_pretrained(quality.MODEL, local_files_only=True)
    generator = load('shared_map_synthetic_check', OLD / 'protocol.py')
    rows = manifest['records']
    assert len(rows) == 3 and len({r['input_ids_sha256'] for r in rows}) == 3
    for row, spec in zip(rows, specs()):
        assert all(row[k] == v for k,v in spec.items())
        expected = generator.make_prompt(spec, lambda p: encode(tok,p))
        ids = expected.pop('ids')
        assert json.loads(json.dumps(expected)) == {k:row[k] for k in expected}
        path = HERE / 'inputs' / (row['id']+'.ids')
        assert path.read_bytes() == struct.pack('<'+'q'*len(ids), *ids)
        assert sha(path) == row['input_ids_sha256'] and row['input_tokens'] == len(ids)
    print('PREFLIGHT PASS', flush=True)
    return quality, manifest, tok


def repetition(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    fraction = 1-len(set(grams))/len(grams) if grams else 0.0
    tail = ids[-32:]
    periods = [p for p in range(1,9) if len(tail)==32 and all(tail[i]==tail[i%p] for i in range(32))]
    return dict(repeated_4gram_fraction=fraction, suffix_period=min(periods) if periods else None,
                flagged=len(ids)>=32 and (bool(periods) or fraction>=0.8))


def validate_rows(rows, records):
    expected = {r['id']:r for r in records}
    assert len(rows) == 6 and {(r['id'],r['method']) for r in rows} == {
        (i,m) for i in expected for m in METHODS}
    for row in rows:
        item = expected[row['id']]
        assert row['input_ids_sha256'] == item['input_ids_sha256']
        assert row['correct'] == int(row['prediction'].strip().strip('"\'`').strip()==item['answer'])
        assert 1<=len(row['generated_ids'])<=128
        assert row['full_kv_cache'] and row['finite_all_steps'] and row['dense_decode']
        assert set(map(int,row['prefill'])) == set(range(0,32,2))
        n = item['input_tokens']
        for stat in row['prefill'].values():
            assert stat['full_maps_computed']==32 and stat['score_elements']==32*n*n
            assert stat['full_causal_pairs']==32*n*(n+1)//2
            assert stat['rows_per_head']==[[n//32+int(h<n%32)] for h in range(32)]
            assert stat['max_row_sum_error']<=2e-5
            assert stat['stitched_causal_pairs']==(n*(n+1)//2 if row['method']=='shared_map' else None)
        assert row['decode_calls']=={str(i):len(row['generated_ids'])-1 for i in range(0,32,2)}


def generate(model, tok, item, method, deadline):
    import numpy as np
    import torch
    from transformers import DynamicCache
    from adapter import install
    state = install(model, method)
    cache = DynamicCache()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else list(eos)
    ids = np.fromfile(HERE/'inputs'/(item['id']+'.ids'), dtype='<i8').tolist()
    def forward(tokens, offset):
        deadline()
        with torch.no_grad():
            out = model.model(input_ids=tokens, use_cache=True, past_key_values=cache,
                position_ids=torch.arange(offset,offset+tokens.shape[1],device='cuda')[None],
                attention_mask=torch.ones(1,offset+tokens.shape[1],dtype=torch.long,device='cuda'),
                return_dict=True)
            logits = model.lm_head(out.last_hidden_state[:,-1]).float()[0]
        assert bool(torch.isfinite(logits).all())
        assert len(cache.key_cache)==len(cache.value_cache)==32
        assert all(k.shape==v.shape==(1,8,offset+tokens.shape[1],128)
                   for k,v in zip(cache.key_cache,cache.value_cache))
        return logits
    start = time.monotonic()
    try:
        logits = forward(torch.tensor([ids],device='cuda'),0)
        torch.cuda.synchronize()
        prefill_seconds = time.monotonic()-start
        first = logits.cpu().numpy().copy()
        generated = []
        for j in range(128):
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j==127:
                break
            logits = forward(torch.tensor([[token]],device='cuda'),len(ids)+j)
        torch.cuda.synchronize()
        prediction = tok.decode(generated, skip_special_tokens=True)
        return dict(id=item['id'],method=method,input_ids_sha256=item['input_ids_sha256'],
            generated_ids=generated,prediction=prediction,answer=item['answer'],
            correct=int(prediction.strip().strip('"\'`').strip()==item['answer']), eos_ids=eos,
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',
            full_kv_cache=True, finite_all_steps=True, dense_decode=True,
            prefill={str(k):v for k,v in state['prefill'].items()},
            decode_calls={str(i):state['decode_calls'].get(i,0) for i in state['layers']},
            repetition=repetition(generated),prefill_seconds=prefill_seconds,
            total_seconds=time.monotonic()-start), first
    finally:
        state['close']()
        del cache
        gc.collect()
        torch.cuda.empty_cache()


def run():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    quality, manifest, tok = check()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert 'H100' in torch.cuda.get_device_name() and transformers.__version__.startswith('4.44.')
    assert (HERE/'submitted-job.txt').read_text().strip().split(';')[0]==os.environ['SLURM_JOB_ID']
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    start = time.monotonic()
    def deadline():
        assert time.monotonic()-start<1560, 'Smoke budget guard; no automatic retry'
    (HERE/'results').mkdir()
    save(HERE/'run-start.json',dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(), manifest_sha256=sha(HERE/'inputs/manifest.json'),
        source_manifest_sha256=sha(HERE/'SHA256SUMS'),torch=torch.__version__,transformers=transformers.__version__))
    model = AutoModelForCausalLM.from_pretrained(quality.MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    quality.install_mlp(model)  # Existing frozen FFN memory implementation, not a learned corrector.
    assert (model.config.num_hidden_layers,model.config.num_attention_heads,
            model.config.num_key_value_heads,model.config.hidden_size)==(32,32,8,4096)
    versions = {name:p._version for name,p in model.named_parameters()}
    rows=[]
    for item in manifest['records']:
        for method in METHODS:
            row,logits = generate(model,tok,item,method,deadline)
            path=HERE/'results'/f'{item["id"]}-{method}.npy'
            with path.open('xb') as f:
                np.save(f,logits,allow_pickle=False)
            row.update(logits_file=path.name,logits_sha256=sha(path))
            save(path.with_suffix('.json'),row)
            rows.append(row)
            print('RESULT',item['id'],method,row['correct'],row['stop_reason'],repr(row['prediction']),flush=True)
            if len(rows)==2:
                elapsed=time.monotonic()-start
                projection=elapsed+max(r['total_seconds'] for r in rows)*4*1.5+120
                save(HERE/'budget-projection.json',dict(projected_seconds=projection,limit_seconds=1560))
                assert projection<1560, 'Budget projection rejected; preserve outputs, no retry'
    validate_rows(rows,manifest['records'])
    assert versions=={name:p._version for name,p in model.named_parameters()}
    save(HERE/'complete.json',dict(status='complete',records=6,manifest_sha256=sha(HERE/'inputs/manifest.json'),
        frozen_base_unchanged=True,summary={m:dict(correct=sum(r['correct'] for r in rows if r['method']==m),
        count=3,caps=sum(r['stop_reason']=='max_new_tokens' for r in rows if r['method']==m),
        repetitive=sum(r['repetition']['flagged'] for r in rows if r['method']==m)) for m in METHODS},
        results={p.name:sha(p) for p in sorted((HERE/'results').glob('*'))},elapsed_seconds=time.monotonic()-start))
    print('SHARED MAP COMPLETE6',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','check','run'))
    args=parser.parse_args()
    try:
        globals()[args.action]()
    except Exception as exc:
        if args.action=='run' and not (HERE/'failure.json').exists():
            save(HERE/'failure.json',dict(type=type(exc).__name__,message=str(exc),job=os.environ.get('SLURM_JOB_ID')))
        raise
