"""Smoke-gated, isolated two-shard-per-method experiment. Never overwrites attempts."""
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
import yaml

HERE=Path(__file__).resolve().parent
WORK=Path('/home/ubuntu/work')
ROOT=WORK/'experiments/outputs/llama_kv_probmean_compact_20260906'
PY='/home/ubuntu/miniconda3/envs/attentionmap/bin/python'
REFERENCE=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl'
GROUP=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json'
DATA=Path('/local/experiment-data/infinitebench_benchmark_specific_calibration/filtered_data')
CONFIG=DATA.parent/'task_configs'
METHODS=['probmean_shared_compact10240','probmean_perhead_compact10240','probmean_shared_compact_topp99','probmean_perhead_compact_topp99']
CONTROL='logitmean_notail_shared_compact10240'

def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def identity(r): return (r['input_ids_sha256'],r['input_tokens'])
def status(**kw):
    with (ROOT/'scheduler_status.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),**kw})+'\n')

def prepare(a,b):
    dest=ROOT/'data'/f'{a}_{b}'
    if (dest/'task_configs/kv_retrieval.yaml').exists(): return dest/'task_configs'
    lines=(DATA/'kv_retrieval.jsonl').read_text().splitlines(keepends=True)
    assert len(lines)==497
    (dest/'filtered_data').mkdir(parents=True,exist_ok=True)
    (dest/'task_configs').mkdir(exist_ok=True)
    (dest/'filtered_data/kv_retrieval.jsonl').write_text(''.join(lines[a:b]))
    node=yaml.compose((CONFIG/'kv_retrieval.yaml').read_text())
    changed=[]
    def visit(n):
        if isinstance(n,yaml.ScalarNode) and n.value==str(DATA):
            n.value=str(dest/'filtered_data'); changed.append(1)
        elif isinstance(n,yaml.MappingNode):
            for k,v in n.value: visit(k); visit(v)
        elif isinstance(n,yaml.SequenceNode):
            for x in n.value: visit(x)
    visit(node); assert len(changed)==1
    (dest/'task_configs/kv_retrieval.yaml').write_text(yaml.serialize(node))
    for p in CONFIG.glob('*.py'): shutil.copy2(p,dest/'task_configs'/p.name)
    return dest/'task_configs'

def validate(directory,count):
    s=json.loads((directory/'summary.json').read_text())
    rr=rows(directory/'online_metrics.jsonl'); ref=rows(REFERENCE)
    ids=[identity(r) for r in rr]; refids=[identity(r) for r in ref]
    assert len(rr)==count and len(set(ids))==count
    assert all(i in set(refids) for i in ids)
    assert [i for i in refids if i in set(ids)]==ids
    assert s['input_alignment']['status']=='passed'
    assert s['method']==directory.parent.name
    if s['method']==CONTROL:
        assert s['method_metadata']['online_config'].get('selection_rank')=='raw_logits'
    assert s['run_args']['max_new_tokens']==128
    samples=json.loads((directory/'lm_eval_results.json').read_text())['samples']['kv_retrieval']
    assert len(samples)==count
    scores=[x['get_score_one_kv_retrieval'] for x in samples]
    assert all(x in (0,1) for x in scores)
    assert abs(sum(scores)/count-s['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none'])<1e-9
    assert all(r['max_probability_sum_error']<1e-4 for r in rr)
    return s,rr

def validate_smoke_arrays(directory):
    for row in rows(directory/'online_metrics.jsonl'):
        with np.load(row['selection_stats_file']) as arrays:
            for array in arrays.values():
                ends=np.minimum((np.arange(len(array))+1)*128,row['input_tokens'])
                assert np.isfinite(array).all()
                assert np.abs(array[:,4]-1).max()<1e-4
                assert (array[:,1]<=ends).all()
                if 'topp99' in str(directory):
                    assert (array[:,2]>=.99-1e-4).all()
                    assert (array[:,0]>=np.minimum(ends,1024)).all()
                    assert (array[:,1]>=array[:,0]).all()
                else:
                    assert np.array_equal(array[:,1],np.minimum(ends,10240))

def run_job(gpu,method,a,b,phase):
    base=ROOT/phase/f'{method}_{a}_{b}'
    base.mkdir(parents=True,exist_ok=True)
    for old in sorted(base.glob('attempt*')):
        d=old/method/'kv_retrieval'
        try: validate(d,b-a); return d
        except (AssertionError,KeyError,FileNotFoundError,json.JSONDecodeError): pass
    config=prepare(a,b)
    # Claim only this GPU and refuse to start if another compute process is present.
    lock=(ROOT/f'gpu{gpu}.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuids[gpu] in active: raise RuntimeError(f'GPU{gpu} occupied; no process was stopped')
    attempt=base/f'attempt{len(list(base.glob("attempt*")))+1}'; attempt.mkdir()
    cmd=[PY,str(HERE/'run.py'),'--model',str(WORK/'model/Llama-3.1-8B-Instruct'),
         '--method',method,'--task','kv_retrieval','--max_length','131072','--batch_size','1',
         '--seed','42','--chat','--group_config',str(GROUP),'--flexprefill_root',str(WORK/'FlexPrefill'),
         '--task_config_dir',str(config),'--record_sparsity','--fixed_topk_budget','10240',
         '--output_dir',str(attempt),'--reference_metrics',str(REFERENCE)]
    (attempt/'command.json').write_text(json.dumps(cmd,indent=2))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(WORK),'TOKENIZERS_PARALLELISM':'false'}
    status(event='start',gpu=gpu,method=method,phase=phase,start=a,end=b,attempt=str(attempt))
    with (attempt/'run.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(event='pid',gpu=gpu,pid=process.pid,method=method,phase=phase)
        code=process.wait()
    if code: raise RuntimeError(f'{method} {phase} failed ({code}): {attempt}/run.log')
    d=attempt/method/'kv_retrieval'; validate(d,b-a)
    status(event='complete',gpu=gpu,method=method,phase=phase,count=b-a)
    return d

def merge(method,dirs):
    dest=ROOT/'merged'/method/'kv_retrieval'
    if (dest/'summary.json').exists():
        s=json.loads((dest/'summary.json').read_text())
        assert s['input_alignment']['status']=='passed' and s['input_alignment']['count']==497
        return
    if dest.exists(): raise RuntimeError(f'Refusing to overwrite incomplete merge: {dest}')
    summaries=[]; merged=[]; predictions=[]
    for d in dirs:
        s,rr=validate(d,len(rows(d/'online_metrics.jsonl'))); summaries.append(s); merged+=rr
        for item in json.loads((d/'lm_eval_results.json').read_text())['samples']['kv_retrieval']:
            predictions.append({**item,'source_shard':str(d)})
    indexed={identity(r):r for r in merged}; ref=rows(REFERENCE)
    assert len(merged)==len(indexed)==len(ref)==497
    rr=[indexed[identity(r)] for r in ref]
    for i,r in enumerate(rr): r['call_index']=i
    selected=sum(r['selected_token_pairs'] for r in rr); causal=sum(r['causal_token_pairs'] for r in rr)
    assert causal==sum(32*32*r['input_tokens']*(r['input_tokens']+1)//2 for r in rr)
    key='get_score_one_kv_retrieval,none'
    correct=round(sum(s['official_lm_eval_results']['kv_retrieval'][key]*s['runtime']['count'] for s in summaries))
    assert len(predictions)==len({x['doc_hash'] for x in predictions})==497
    assert correct==sum(x['get_score_one_kv_retrieval'] for x in predictions)
    timing=np.array([r['prefill_latency_sec'] for r in rr])
    runtime={'count':497,'correct':correct,'accuracy':correct/497,
        'avg_prefill_latency_sec':float(timing.mean()),'median_prefill_latency_sec':float(np.median(timing)),
        'p90_prefill_latency_sec':float(np.quantile(timing,.9)),
        'avg_decode_latency_sec':float(np.mean([r['decode_latency_sec'] for r in rr])),
        'avg_total_latency_sec':float(np.mean([r['total_latency_sec'] for r in rr])),
        'selected_token_pairs':selected,'causal_token_pairs':causal,'global_token_sparsity':1-selected/causal,
        'max_peak_memory_bytes':max(r.get('peak_memory_bytes',r.get('max_peak_memory_bytes',0)) for r in rr)}
    dest.mkdir(parents=True)
    (dest/'online_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    report={**summaries[0],'runtime':runtime,'official_lm_eval_results':{'kv_retrieval':{key:correct/497}},
        'input_alignment':{'status':'passed','alignment_scope':'full','count':497,'reference_count':497,
                           'reference_metrics':str(REFERENCE),'compared_fields':['input_ids_sha256','input_tokens']},
        'source_shards':[str(d) for d in dirs]}
    from summarize import aggregate
    aggregate(dest)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['smoke','formal','all'],default='all'); args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'scheduler.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'correctness.json').read_text())['passed']
    for a,b in [(0,3),(0,249),(249,497)]: prepare(a,b)
    if args.phase in ('smoke','all'):
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(run_job,g,m,0,3,'smoke') for g,m in enumerate(METHODS+[CONTROL])]
            ds=[f.result() for f in futures]
        identities=[[identity(r) for r in rows(d/'online_metrics.jsonl')] for d in ds]
        assert all(x==identities[0] for x in identities)
        for d in ds: validate_smoke_arrays(d)
        (ROOT/'smoke_passed.json').write_text(json.dumps({'passed':True,'dirs':[str(d) for d in ds]},indent=2))
    if args.phase in ('formal','all'):
        assert json.loads((ROOT/'smoke_passed.json').read_text())['passed']
        for d in json.loads((ROOT/'smoke_passed.json').read_text())['dirs']:
            validate(Path(d),3)
        frozen=ROOT/'formal_source'
        frozen.mkdir(exist_ok=True)
        hashes={}
        for source in HERE.glob('*.py'):
            target=frozen/source.name
            if target.exists() and target.read_bytes()!=source.read_bytes():
                raise RuntimeError(f'Formal source changed: {source}')
            if not target.exists(): shutil.copy2(source,target)
            hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
        (ROOT/'formal_source_hashes.json').write_text(json.dumps(hashes,indent=2))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures={m:[pool.submit(run_job,mi*2+si,m,a,b,'formal') for si,(a,b) in enumerate([(0,249),(249,497)])] for mi,m in enumerate(METHODS)}
            for m,fs in futures.items(): merge(m,[f.result() for f in fs])
        from summarize import comparison
        comparison(ROOT)
        (ROOT/'complete.json').write_text(json.dumps({'complete':True,'count':1988,'methods':METHODS},indent=2))

if __name__=='__main__':
    try: main()
    except Exception as e:
        if ROOT.exists(): status(event='scheduler_failed',error=repr(e))
        raise
