"""Single-GPU array worker; paired inputs, bounded runtime, exclusive artifacts."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

from protocol import RUN, PRIOR, TASKS, METHODS, LAYERS, MAX_NEW, SEED
from protocol import sha, digest, save, official, render, validate_inputs, validate_pair, repetition
from prepare import quality


def check(retokenize=False):
    import numpy as np
    from transformers import AutoTokenizer
    assert Path(__file__).resolve().parent == RUN
    for line in (RUN/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split()
        assert sha(RUN/name) == expected, name
    assert sha(PRIOR/'env.sh') == 'c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9'
    q, prior = quality()
    m = json.loads((RUN/'inputs/manifest.json').read_text())
    validate_inputs(m)
    assert m['tokenizer_files'] == prior['tokenizer_files']
    assert m['model'] == str(q.MODEL)
    assert sha(q.MODEL/'config.json') == m['model_config_sha256']
    assert sha(q.MODEL/'generation_config.json') == m['generation_config_sha256']
    for name, value in m['preparation_sources'].items():
        assert sha(RUN/name) == sha(RUN/'source-preparation'/name) == value
    for name, value in m['official_hashes'].items():
        assert sha(RUN/'official'/name) == value
    for task, info in m['datasets'].items():
        assert sha(RUN/'data'/info['file']) == info['sha256']
        lines = (RUN/'data'/info['file']).read_text().splitlines()
        selection = json.loads((RUN/'data'/(task+'-selection.json')).read_text())
        order = list(range(len(lines)))
        random.Random(SEED+TASKS.index(task)).shuffle(order)
        scanned = selection['scanned']
        assert [s['source_index'] for s in scanned] == order[:len(scanned)]
        assert selection['chosen'] == [s['source_index'] for s in scanned if s['eligible']]
        assert len(selection['chosen']) == 50
        for item in (r for r in m['records'] if r['task']==task):
            assert item['source_index'] in selection['chosen']
            assert (RUN/'inputs'/(item['key']+'.raw')).read_bytes() == lines[item['source_index']].encode()
        del lines
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True)
    templates, get_answer, scorers = official(RUN/'official')
    fields = {'ids':'input_ids_sha256', 'pos':'position_ids_sha256',
              'mask':'attention_mask_sha256', 'prompt':'rendered_prompt_sha256', 'raw':'raw_row_sha256'}
    for item in m['records']:
        for suffix, field in fields.items():
            assert sha(RUN/'inputs'/(item['key']+'.'+suffix)) == item[field]
        p = RUN/'inputs'/item['key']
        ids = np.fromfile(str(p)+'.ids', dtype='<i8')
        pos = np.fromfile(str(p)+'.pos', dtype='<i8')
        mask = np.fromfile(str(p)+'.mask', dtype='u1')
        assert len(ids) == len(pos) == len(mask) == item['input_tokens']
        assert np.array_equal(pos, np.arange(len(ids))) and (mask == 1).all()
        row = json.loads(Path(str(p)+'.raw').read_text())
        answers = get_answer(row, item['task'])
        assert (answers if isinstance(answers,list) else [answers]) == item['answers']
        if retokenize:
            rendered = tok.apply_chat_template([dict(role='user',content=render(row,item['task'],templates))],
                tokenize=False, add_generation_prompt=True)
            assert rendered == Path(str(p)+'.prompt').read_text()
            assert tok.encode(rendered,add_special_tokens=False) == ids.tolist()
    config = dict(manifest_sha256=sha(RUN/'inputs/manifest.json'),
        source_manifest_sha256=sha(RUN/'SHA256SUMS'), methods=list(METHODS), layers=list(LAYERS),
        seed=42, query_block_size=256, dtype='BF16', attention_arithmetic='FP32 exact all-head full-key softmax',
        prefill_only=True, full_kv=True, dense_decode=True, training=False, max_new_tokens=MAX_NEW,
        budget=dict(workers=4,gpu='H10080GB',gpus_per_worker=1,cpus=8,memory_gb=128,max_seconds=86400))
    print('FROZEN PREFLIGHT PASS200', config['manifest_sha256'], flush=True)
    return q, m, tok, scorers, config


def gpu_parity():
    import torch
    from attention import full_attention
    from reference_attention import full_attention as reference
    torch.manual_seed(931)
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(1,32,35,128,device='cuda',dtype=dtype)
        k = torch.randn(1,8,35,128,device='cuda',dtype=dtype)
        v = torch.randn_like(k)
        pos = torch.arange(35,device='cuda')[None]
        for mode in METHODS:
            actual, amap, _ = full_attention(q,k,v,pos,mode=mode,block_size=7,return_map=True)
            expected, emap, _ = reference(q,k,v,pos,mode=mode)
            torch.testing.assert_close(actual,expected,atol=0.02 if dtype==torch.bfloat16 else 2e-5,rtol=0.01 if dtype==torch.bfloat16 else 2e-5)
            if amap is not None:
                torch.testing.assert_close(amap,emap,atol=2e-6,rtol=2e-5)
    print('GPU BLOCK/LITERAL PARITY PASS',flush=True)


def generate(model,tok,item,method,deadline,output,metadata,scorer):
    import numpy as np
    import torch
    from transformers import DynamicCache
    from adapter import install
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    source = RUN/'inputs'/item['key']
    assert sha(str(source)+'.ids') == item['input_ids_sha256']
    ids = np.fromfile(str(source)+'.ids',dtype='<i8').tolist()
    pos = np.fromfile(str(source)+'.pos',dtype='<i8').tolist()
    mask = np.fromfile(str(source)+'.mask',dtype='u1').tolist()
    state = install(model,method)
    cache = DynamicCache()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos,int) else list(eos)
    assert eos
    def forward(tokens,offset):
        deadline()
        n = offset+tokens.shape[1]
        positions = torch.tensor([pos],device='cuda') if offset==0 else torch.tensor([[offset]],device='cuda')
        masks = torch.tensor([mask],device='cuda') if offset==0 else torch.ones(1,n,dtype=torch.long,device='cuda')
        with torch.no_grad():
            out = model.model(input_ids=tokens,position_ids=positions,attention_mask=masks,
                use_cache=True,past_key_values=cache,return_dict=True)
            logits = model.lm_head(out.last_hidden_state[:,-1]).float()[0]
        assert bool(torch.isfinite(logits).all())
        assert len(cache.key_cache)==len(cache.value_cache)==32
        assert all(k.shape==v.shape==(1,8,n,128) for k,v in zip(cache.key_cache,cache.value_cache))
        return logits
    start = time.monotonic()
    try:
        logits = forward(torch.tensor([ids],device='cuda'),0)
        torch.cuda.synchronize()
        prefill_seconds = time.monotonic()-start
        first = logits.cpu().numpy().copy()
        generated = []
        for j in range(item['max_new_tokens']):
            token = int(logits.argmax())
            generated.append(token)
            if token in eos or j==item['max_new_tokens']-1:
                break
            logits = forward(torch.tensor([[token]],device='cuda'),len(ids)+j)
        torch.cuda.synchronize()
        total = time.monotonic()-start
        prediction = tok.decode(generated,skip_special_tokens=True)
        path = output/(item['key']+'-'+method+'.npy')
        with path.open('xb') as f:
            np.save(f,first,allow_pickle=False)
        for stat in state['prefill'].values():
            n = len(ids)
            assert stat['stitched_causal_pairs'] == (n*(n+1)//2 if method=='shared_map' else None)
        row = dict(item, **metadata, method=method,generated_ids=generated,prediction=prediction,
            score=float(scorer(prediction,item['answers'],'llama')),eos_ids=eos,
            stop_reason='eos' if generated[-1] in eos else 'max_new_tokens',full_kv_cache=True,
            finite_all_steps=True,dense_decode=True,prefill={str(k):v for k,v in state['prefill'].items()},
            decode_calls={str(i):state['decode_calls'].get(i,0) for i in LAYERS},
            repetition=repetition(generated),prefill_seconds=prefill_seconds,total_seconds=total,
            decode_seconds=total-prefill_seconds,logits_file=path.name,logits_sha256=sha(path))
        save(path.with_suffix('.json'),row)
        return row
    finally:
        state['close']()
        del cache
        gc.collect()
        torch.cuda.empty_cache()


def projection(rows, remaining, elapsed):
    estimate = 0.0
    for method in METHODS:
        observed = [r for r in rows if r['method']==method]
        prefill_rate = max(r['prefill_seconds']/r['input_tokens']**2 for r in observed)
        decode_rate = max(r['decode_seconds']/max(1,len(r['generated_ids'])-1) for r in observed)
        estimate += sum(prefill_rate*r['input_tokens']**2+decode_rate*(r['max_new_tokens']-1) for r in remaining)
    return elapsed+1.5*estimate+1800


def run():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    start = time.monotonic()
    shard = int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert shard in range(4)
    for _ in range(30):
        if (RUN/'submitted-job.txt').exists():
            break
        time.sleep(1)
    assert (RUN/'submitted-job.txt').read_text().strip().split(';')[0] == os.environ['SLURM_ARRAY_JOB_ID']
    output = RUN/f'worker-{shard}'
    output.mkdir()
    q,m,tok,scorers,config = check()
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    assert torch.cuda.get_device_properties(0).total_memory > 75*1024**3
    assert transformers.__version__.startswith('4.44.')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    metadata = dict(manifest_sha256=config['manifest_sha256'],config_sha256=digest(config),
        source_manifest_sha256=config['source_manifest_sha256'],job=os.environ['SLURM_JOB_ID'],
        array_job=os.environ['SLURM_ARRAY_JOB_ID'],node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,generation_seed=42)
    save(output/'run-start.json',dict(metadata,config=config))
    def deadline():
        assert time.monotonic()-start < 84600, 'Runtime guard: stop, preserve, no retry'
    gpu_parity()
    model = AutoModelForCausalLM.from_pretrained(q.MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    q.install_mlp(model)  # Frozen chunked FFN implementation, not a learned correction MLP.
    assert (model.config.num_hidden_layers,model.config.num_attention_heads,model.config.num_key_value_heads,model.config.hidden_size)==(32,32,8,4096)
    versions = {name:p._version for name,p in model.named_parameters()}
    items = sorted((r for r in m['records'] if r['shard']==shard),key=lambda r:(-r['input_tokens']**2,r['key']))
    assert len(items)==50
    rows=[]
    for index,item in enumerate(items):
        pair=[]
        for method in METHODS:
            row = generate(model,tok,item,method,deadline,output,metadata,scorers[item['task']])
            pair.append(row)
            rows.append(row)
            print('RESULT',index,item['key'],method,row['score'],row['stop_reason'],repr(row['prediction']),flush=True)
        validate_pair(pair,item,metadata['manifest_sha256'],metadata['config_sha256'])
        if index>=1:
            projected = projection(rows,items[index+1:],time.monotonic()-start)
            save(output/f'projection-{index:02d}.json',dict(projected_seconds=projected,limit_seconds=84600,completed_pairs=index+1))
            assert projected<84600, 'Budget projection rejected: preserve artifacts, no automatic retry'
            if index==1:
                save(output/'smoke-complete.json',dict(status='passed',records=4,projected_seconds=projected,limit_seconds=84600))
                print('SMOKE PASS; AUTOMATIC CONTINUATION',projected,flush=True)
    assert versions=={name:p._version for name,p in model.named_parameters()}
    save(output/'complete.json',dict(status='complete',records=100,frozen_base_unchanged=True,
        elapsed_seconds=time.monotonic()-start,manifest_sha256=metadata['manifest_sha256'],config_sha256=metadata['config_sha256'],
        files={p.name:sha(p) for p in sorted(output.glob('*')) if p.is_file()},
        summary={t:{method:dict(count=sum(r['task']==t and r['method']==method for r in rows),
            score_sum=sum(r['score'] for r in rows if r['task']==t and r['method']==method)) for method in METHODS} for t in TASKS}))
    print('WORKER COMPLETE100',shard,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','run'))
    args=parser.parse_args()
    try:
        check(retokenize=True) if args.action=='check' else run()
    except Exception as exc:
        if args.action=='run':
            path=RUN/f"worker-{os.environ.get('SLURM_ARRAY_TASK_ID','unknown')}"
            path.mkdir(exist_ok=True)
            if not (path/'failure.json').exists():
                save(path/'failure.json',dict(type=type(exc).__name__,message=str(exc),job=os.environ.get('SLURM_JOB_ID')))
        raise
