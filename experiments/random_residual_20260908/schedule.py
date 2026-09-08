"""Correctness-gated three-input smoke and eight disjoint formal shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments.rep_residual_20260907.prepare import configuration,digest,DATA

HERE=Path(__file__).resolve().parent
ROOT=Path('/local/results/random_residual_allmembers_20260908')
WORK=Path('/home/ubuntu/work')
PY='/home/ubuntu/miniconda3/envs/attentionmap/bin/python'
METHOD='probmean_shared9216_random1024_allmembers_seed42'
REFERENCE=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl'
BASELINE=WORK/'experiments/outputs/llama_kv_probmean_compact_20260906/merged/probmean_shared_compact10240/kv_retrieval'
GROUP=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json'
EDGES=[0,63,125,187,249,311,373,435,497]
SHARDS=list(zip(EDGES[:-1],EDGES[1:]))


def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def identity(r):return r['input_ids_sha256'],r['input_tokens']
def status(**entry):
    with (ROOT/'scheduler_status.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**entry})+'\n')


def setup():
    data=rows(DATA/'kv_retrieval.jsonl');assert len(data)==497
    manifest={'method':METHOD,'seed':42,'model':str(WORK/'model/Llama-3.1-8B-Instruct'),
        'source_data':str(DATA/'kv_retrieval.jsonl'),'data_sha256':hashlib.sha256((DATA/'kv_retrieval.jsonl').read_bytes()).hexdigest(),
        'input_reference':str(REFERENCE),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'group_config':str(GROUP),'group_sha256':hashlib.sha256(GROUP.read_bytes()).hexdigest(),
        'calibrated_residual_config_used':False,'random_members_each_layer':29,'representatives_unchanged':3,
        'smoke_source_indices':[0,1,2],'shards':SHARDS}
    p=ROOT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(manifest))
    else:p.write_text(json.dumps(manifest,indent=2))
    configuration(ROOT/'data'/'smoke',data[:3])
    for a,b in SHARDS:configuration(ROOT/'data'/f'{a}_{b}',data[a:b])
    frozen=ROOT/'source';frozen.mkdir(exist_ok=True)
    hashes={}
    for source in HERE.glob('*.py'):
        target=frozen/source.name
        if target.exists():assert target.read_bytes()==source.read_bytes(),f'Source changed {source}'
        else:shutil.copy2(source,target)
        hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    for relative in ('experiments/probmean_compact_20260906/selector.py',
                     'experiments/probmean_compact_20260906/run.py',
                     'experiments/rep_residual_20260907/policy.py',
                     'experiments/token_compacted_sparse.py','experiments/run_dense_multimodel_infinitebench.py'):
        source=WORK/relative;target=frozen/'dependencies'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:shutil.copy2(source,target)
        hashes[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
    (ROOT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


def validate(directory,count):
    doc=json.loads((directory/'summary.json').read_text())
    rr=rows(directory/'online_metrics.jsonl');ref=rows(REFERENCE)
    got=[identity(r) for r in rr];refids=[identity(r) for r in ref]
    assert len(got)==len(set(got))==count
    assert [x for x in refids if x in set(got)]==got
    assert doc['input_alignment']['status']=='passed'
    assert doc['method']==METHOD and doc['run_args']['max_new_tokens']==128
    config=doc['method_metadata']['online_config']
    assert config['random_members_per_layer']==29 and config['representatives_per_layer']==3
    assert config['offline_residual_calibration'] is False and config['seed']==42
    baseline={identity(r):r for r in rows(BASELINE/'online_metrics.jsonl')}
    for r in rr:
        old=baseline[identity(r)]
        assert r['selected_token_pairs']==old['selected_token_pairs']
        assert r['causal_token_pairs']==old['causal_token_pairs']
        assert r['max_probability_sum_error']<1e-4
    samples=json.loads((directory/'lm_eval_results.json').read_text())['samples']['kv_retrieval']
    assert len(samples)==count
    correct=sum(x['get_score_one_kv_retrieval'] for x in samples)
    assert abs(correct/count-doc['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none'])<1e-9
    return doc,rr,samples


def job(gpu,a,b,phase):
    folder=ROOT/'attempts'/phase/f'{a}_{b}';folder.mkdir(parents=True,exist_ok=True)
    for attempt in sorted(folder.glob('attempt*')):
        p=attempt/METHOD/'kv_retrieval'
        try:validate(p,b-a);return p
        except (FileNotFoundError,AssertionError,KeyError,json.JSONDecodeError):pass
    lock=(ROOT/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuids[gpu] in active:raise RuntimeError(f'GPU{gpu} occupied; not stopping any process')
    attempt=folder/f'attempt{len(list(folder.glob("attempt*")))+1}';attempt.mkdir()
    config=ROOT/'data'/('smoke' if phase=='smoke' else f'{a}_{b}')/'task_configs'
    cmd=[PY,str(HERE/'online.py'),'--model',str(WORK/'model/Llama-3.1-8B-Instruct'),
         '--method',METHOD,'--task','kv_retrieval','--max_length','131072','--batch_size','1',
         '--seed','42','--chat','--group_config',str(GROUP),'--flexprefill_root',str(WORK/'FlexPrefill'),
         '--task_config_dir',str(config),'--record_sparsity','--fixed_topk_budget','10240',
         '--output_dir',str(attempt),'--reference_metrics',str(REFERENCE)]
    env={**os.environ,'PYTHONPATH':str(WORK),'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'}
    (attempt/'command.json').write_text(json.dumps(cmd,indent=2))
    with (attempt/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(event='start',phase=phase,gpu=gpu,pid=proc.pid,a=a,b=b,attempt=str(attempt))
        code=proc.wait()
    if code:raise RuntimeError(f'Inference failed code={code}: {attempt}/run.log')
    p=attempt/METHOD/'kv_retrieval';validate(p,b-a)
    status(event='complete',phase=phase,a=a,b=b)
    return p


def merge(paths):
    dest=ROOT/'merged'/METHOD/'kv_retrieval'
    if dest.exists():raise RuntimeError('Refusing to overwrite merge directory')
    chunks=[validate(p,b-a) for p,(a,b) in zip(paths,SHARDS)]
    unordered=[r for _,rr,_ in chunks for r in rr];indexed={identity(r):r for r in unordered}
    reference=rows(REFERENCE)
    assert len(indexed)==len(unordered)==len(reference)==497
    assert set(indexed)=={identity(r) for r in reference}
    rr=[indexed[identity(r)] for r in reference]
    samples=[dict(r,source_shard=str(p)) for p,(_,_,ss) in zip(paths,chunks) for r in ss]
    assert len(samples)==len({r['doc_hash'] for r in samples})==497
    correct=sum(r['get_score_one_kv_retrieval'] for r in samples)
    selected=sum(r['selected_token_pairs'] for r in rr);causal=sum(r['causal_token_pairs'] for r in rr)
    old=json.loads((BASELINE/'summary.json').read_text())['runtime']
    assert selected==old['selected_token_pairs'] and causal==old['causal_token_pairs']
    prefill=np.array([r['prefill_latency_sec'] for r in rr])
    runtime={'count':497,'correct':correct,'accuracy':correct/497,
        'avg_prefill_latency_sec':float(prefill.mean()),'median_prefill_latency_sec':float(np.median(prefill)),
        'p90_prefill_latency_sec':float(np.quantile(prefill,.9)),
        'avg_decode_latency_sec':float(np.mean([r['decode_latency_sec'] for r in rr])),
        'avg_total_latency_sec':float(np.mean([r['total_latency_sec'] for r in rr])),
        'selected_token_pairs':selected,'causal_token_pairs':causal,'global_token_sparsity':1-selected/causal,
        'max_peak_memory_bytes':max(r.get('peak_memory_bytes',0) for r in rr)}
    for i,r in enumerate(rr):r['call_index']=i
    doc={**chunks[0][0],'runtime':runtime,'source_shards':[str(p) for p in paths],
        'official_lm_eval_results':{'kv_retrieval':{'get_score_one_kv_retrieval,none':correct/497}},
        'input_alignment':{'status':'passed','alignment_scope':'full','count':497,'reference_count':497,
            'reference_metrics':str(REFERENCE),'compared_fields':['input_ids_sha256','input_tokens']},
        'pair_budget_alignment':{'passed':True,'baseline':str(BASELINE),'per_sample_exact':True,'total_exact':True}}
    dest.mkdir(parents=True)
    (dest/'online_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    (dest/'summary.json').write_text(json.dumps(doc,indent=2))
    return dest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.detach:
        logfile=ROOT/f'scheduler_{int(time.time())}.log'
        with logfile.open('w') as f:
            p=subprocess.Popen([PY,str(HERE/'schedule.py')],cwd=WORK,env={**os.environ,'PYTHONPATH':str(WORK)},
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':p.pid,'log':str(logfile)}));return
    lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'correctness.json').read_text())['passed']
    setup();status(event='scheduler_started',pid=os.getpid())
    smoke=job(0,0,3,'smoke')
    _,records,_=validate(smoke,3)
    previous=WORK/'experiments/outputs/llama_kv_probmean_compact_20260906/data/0_3/filtered_data/kv_retrieval.jsonl'
    assert rows(previous)==rows(ROOT/'data/smoke/filtered_data/kv_retrieval.jsonl')
    for record in records:
        p=Path(record['selection_stats_file']);meta=json.loads(p.with_suffix('.json').read_text())
        assert len(meta['mapping'])==1024
        with np.load(p) as arrays:
            for item in meta['mapping']:
                a=arrays[item['array']];ends=np.minimum((np.arange(len(a))+1)*128,record['input_tokens'])
                b=np.minimum(10240,ends);is_rep=item['source_head']==item['member_heads'][0]
                assert np.array_equal(a[:,1],b)
                core=b if is_rep else np.minimum(9216,b)
                assert np.array_equal(a[:,11],core) and np.array_equal(a[:,12],b-core)
    (ROOT/'smoke_passed.json').write_text(json.dumps({'passed':True,'count':3,'directory':str(smoke),
        'full_generation':True,'input_alignment':'passed/ordered_subsequence','pair_budget_exact':True},indent=2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(job,g,a,b,'formal') for g,(a,b) in enumerate(SHARDS)]
        paths=[f.result() for f in futures]
    dest=merge(paths)
    from report import summarize
    summarize(ROOT,dest)
    (ROOT/'complete.json').write_text(json.dumps({'complete':True,'count':497,'alignment':'passed/full','pair_budget_exact':True}))
    status(event='complete_all',count=497)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if ROOT.exists():status(event='scheduler_failed',error=repr(e))
        raise
