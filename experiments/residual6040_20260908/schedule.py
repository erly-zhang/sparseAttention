"""Smoke-gated A/B full generation, then independent three-input profiling."""
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
from experiments.rep_residual_20260907.prepare import configuration,DATA

HERE=Path(__file__).resolve().parent
ROOT=Path('/local/results/residual6040_20260908')
WORK=Path('/home/ubuntu/work');PY='/home/ubuntu/miniconda3/envs/attentionmap/bin/python'
METHODS=['probmean_shared6144_random4096_allmembers_seed42','probmean_shared6144_member2q4096_allmembers']
MODEL=WORK/'model/Llama-3.1-8B-Instruct'
REFERENCE=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl'
BASELINE=WORK/'experiments/outputs/llama_kv_probmean_compact_20260906/merged/probmean_shared_compact10240/kv_retrieval'
GROUP=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json'
SHARDS=[(0,125),(125,250),(250,375),(375,497)]


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def identity(r):return r['input_ids_sha256'],r['input_tokens']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**x):
    with (ROOT/'scheduler_status.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**x})+'\n')


def setup():
    model=read(MODEL/'config.json')
    assert model['num_attention_heads']==32 and model['num_key_value_heads']==8 and model['num_hidden_layers']==32
    weights=read(MODEL/'model.safetensors.index.json')['weight_map']
    assert all((MODEL/name).is_file() for name in set(weights.values()))
    ref=rows(REFERENCE);assert len(ref)==len(set(identity(r) for r in ref))==497
    baseline=rows(BASELINE/'online_metrics.jsonl')
    assert set(map(identity,ref))==set(map(identity,baseline))
    groups=read(GROUP)['layers'];assert len(groups)==32
    for g in groups.values():assert len(g)==3 and sorted(h for s in g for h in s['members'])==list(range(32))
    data=rows(DATA/'kv_retrieval.jsonl');assert len(data)==497
    previous=WORK/'experiments/outputs/llama_kv_probmean_compact_20260906/data/0_3/filtered_data/kv_retrieval.jsonl'
    assert data[:3]==rows(previous)
    manifest={'model':str(MODEL),'model_config_sha256':sha(MODEL/'config.json'),
        'model_index_sha256':sha(MODEL/'model.safetensors.index.json'),'model_weight_files':sorted(set(weights.values())),
        'reference':str(REFERENCE),'reference_sha256':sha(REFERENCE),'data':str(DATA/'kv_retrieval.jsonl'),
        'data_sha256':sha(DATA/'kv_retrieval.jsonl'),'group_config':str(GROUP),'group_sha256':sha(GROUP),
        'methods':METHODS,'seed':42,'formal_count_per_method':497,'smoke_source_indices':[0,1,2],
        'profile_source_indices':[0,1,2],'shards':SHARDS,'representatives':3,'members':29,
        'no_offline_residual_calibration':True,'no_production_source_edits':True,
        'prefill_scope':'synchronized selector + short-index construction + required actual pair counts + unchanged compact attention',
        'extra_diagnostics':'separate 3-input profiling only'}
    path=ROOT/'manifest.json'
    if path.exists():assert read(path)==json.loads(json.dumps(manifest))
    else:path.write_text(json.dumps(manifest,indent=2))
    configuration(ROOT/'data/smoke',data[:3])
    for a,b in SHARDS:configuration(ROOT/'data'/f'{a}_{b}',data[a:b])
    frozen=ROOT/'source';frozen.mkdir(exist_ok=True);hashes={}
    sources=list(HERE.glob('*.py'))+[HERE/'correctness.json']
    for name in ('experiments/probmean_compact_20260906/selector.py','experiments/probmean_compact_20260906/run.py',
                 'experiments/random_residual_20260908/random_policy.py','experiments/rep_residual_20260907/policy.py',
                 'experiments/rep_residual_20260907/prepare.py','experiments/token_compacted_sparse.py',
                 'experiments/run_dense_multimodel_infinitebench.py','experiments/benchmark_shareprefill_ae3.py'):
        sources.append(WORK/name)
    for source in sources:
        rel=source.relative_to(WORK);target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes(),f'Frozen source differs: {source}'
        else:shutil.copy2(source,target)
        hashes[str(rel)]=sha(source)
    (ROOT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


def validate(path,method,count,profile=False):
    s=read(path/'summary.json');rr=rows(path/'online_metrics.jsonl')
    ids=list(map(identity,rr));ref=list(map(identity,rows(REFERENCE)))
    assert len(ids)==len(set(ids))==count and [x for x in ref if x in set(ids)]==ids
    align=s['input_alignment'];assert align['status']=='passed' and align['count']==count and align['reference_count']==497
    assert s['method']==method and s['run_args']['max_new_tokens']==128
    config=s['method_metadata']['online_config']
    assert config['final_budget']==10240 and config['member_core_budget']==6144 and config['all_nonrepresentatives']==29
    assert s['method_metadata']['profile_only']==profile
    old={identity(r):r for r in rows(BASELINE/'online_metrics.jsonl')}
    for r in rr:
        assert r['profile_only']==profile
        if count==3 and not profile:assert r['smoke_indices_and_finite_logits_passed']
        assert all(np.isfinite(r[k]) and r[k]>0 for k in ('prefill_latency_sec','total_latency_sec'))
        assert r['selected_token_pairs']==old[identity(r)]['selected_token_pairs']
        assert r['causal_token_pairs']==old[identity(r)]['causal_token_pairs']
        if profile:assert Path(r['profiling_file']).is_file()
    samples=read(path/'lm_eval_results.json')['samples']['kv_retrieval']
    assert len(samples)==len({r['doc_hash'] for r in samples})==count
    score=sum(r['get_score_one_kv_retrieval'] for r in samples)/count
    assert abs(score-s['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none'])<1e-9
    return s,rr,samples


def job(gpu,method,a,b,phase):
    directory=ROOT/'attempts'/phase/method/f'{a}_{b}';directory.mkdir(parents=True,exist_ok=True)
    for attempt in sorted(directory.glob('attempt*')):
        p=attempt/method/'kv_retrieval'
        try:validate(p,method,b-a,phase=='profile');return p
        except (FileNotFoundError,AssertionError,KeyError,json.JSONDecodeError):pass
    lock=(ROOT/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuids[gpu] in active:raise RuntimeError(f'GPU{gpu} occupied; not modifying any existing process')
    attempt=directory/f'attempt{len(list(directory.glob("attempt*")))+1}';attempt.mkdir()
    config=ROOT/'data'/('smoke' if phase in ('smoke','profile') else f'{a}_{b}')/'task_configs'
    cmd=[PY,str(HERE/'online.py'),'--model',str(MODEL),'--method',method,'--task','kv_retrieval',
        '--max_length','131072','--batch_size','1','--seed','42','--chat','--group_config',str(GROUP),
        '--flexprefill_root',str(WORK/'FlexPrefill'),'--task_config_dir',str(config),'--record_sparsity',
        '--fixed_topk_budget','10240','--output_dir',str(attempt),'--reference_metrics',str(REFERENCE)]
    env={**os.environ,'PYTHONPATH':str(WORK),'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false',
         'RESIDUAL6040_PROFILE':str(int(phase=='profile')),'RESIDUAL6040_VALIDATE':str(int(phase=='smoke'))}
    (attempt/'command.json').write_text(json.dumps({'command':cmd,'profile':phase=='profile'},indent=2))
    with (attempt/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(event='start',phase=phase,method=method,gpu=gpu,pid=proc.pid,a=a,b=b,attempt=str(attempt))
        code=proc.wait()
    if code:raise RuntimeError(f'{method} {phase} failed code={code}: {attempt}/run.log')
    path=attempt/method/'kv_retrieval';validate(path,method,b-a,phase=='profile')
    status(event='complete',phase=phase,method=method,a=a,b=b)
    return path


def merge(method,paths):
    dest=ROOT/'merged'/method/'kv_retrieval'
    if dest.exists():
        assert read(dest/'summary.json')['input_alignment']['status']=='passed'
        return dest
    chunks=[validate(p,method,b-a) for p,(a,b) in zip(paths,SHARDS)]
    rr=[r for _,rs,_ in chunks for r in rs];by_id={identity(r):r for r in rr};ref=rows(REFERENCE)
    assert len(by_id)==len(rr)==497 and set(by_id)==set(map(identity,ref))
    rr=[by_id[identity(r)] for r in ref]
    samples=[dict(x,source_shard=str(p)) for p,(_,_,ss) in zip(paths,chunks) for x in ss]
    assert len(samples)==len({x['doc_hash'] for x in samples})==497
    correct=sum(x['get_score_one_kv_retrieval'] for x in samples)
    pre=np.array([r['prefill_latency_sec'] for r in rr]);selected=sum(r['selected_token_pairs'] for r in rr);causal=sum(r['causal_token_pairs'] for r in rr)
    old=read(BASELINE/'summary.json')['runtime']
    assert selected==old['selected_token_pairs'] and causal==old['causal_token_pairs']
    runtime={'count':497,'correct':correct,'accuracy':correct/497,'avg_prefill_latency_sec':float(pre.mean()),
        'median_prefill_latency_sec':float(np.median(pre)),'p90_prefill_latency_sec':float(np.quantile(pre,.9)),
        'avg_decode_latency_sec':float(np.mean([r['decode_latency_sec'] for r in rr])),
        'avg_total_latency_sec':float(np.mean([r['total_latency_sec'] for r in rr])),
        'selected_token_pairs':selected,'causal_token_pairs':causal,'global_token_sparsity':1-selected/causal,
        'max_peak_memory_bytes':max(r['peak_memory_bytes'] for r in rr)}
    s={**chunks[0][0],'runtime':runtime,'source_shards':[str(p) for p in paths],
        'input_alignment':{'status':'passed','alignment_scope':'full','count':497,'reference_count':497,
            'reference_metrics':str(REFERENCE),'compared_fields':['input_ids_sha256','input_tokens']},
        'pair_budget_alignment':{'passed':True,'per_sample_exact':True,'total_exact':True,'reference':str(BASELINE)},
        'official_lm_eval_results':{'kv_retrieval':{'get_score_one_kv_retrieval,none':correct/497}}}
    dest.mkdir(parents=True)
    for i,r in enumerate(rr):r['call_index']=i
    (dest/'online_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    (dest/'summary.json').write_text(json.dumps(s,indent=2));return dest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.detach:
        path=ROOT/f'scheduler_{int(time.time())}.log'
        with path.open('w') as log:
            proc=subprocess.Popen([PY,str(HERE/'schedule.py')],cwd=WORK,env={**os.environ,'PYTHONPATH':str(WORK)},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':proc.pid,'log':str(path)}));return
    lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(HERE/'correctness.json')['passed'];setup();status(event='scheduler_start',pid=os.getpid())
    with ThreadPoolExecutor(max_workers=2) as pool:
        smoke=[pool.submit(job,g,m,0,3,'smoke') for g,m in zip((0,4),METHODS)]
        smoke=[x.result() for x in smoke]
    (ROOT/'smoke_passed.json').write_text(json.dumps({'passed':True,'both_methods':True,'full_max_new_tokens':128,
        'count_per_method':3,'paths':list(map(str,smoke)),'input_alignment':'passed/ordered_subsequence','pair_budgets_exact':True},indent=2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[[pool.submit(job,i*4+g,m,a,b,'formal') for g,(a,b) in enumerate(SHARDS)] for i,m in enumerate(METHODS)]
        formal=[[f.result() for f in group] for group in futures]
    merged=[merge(m,pp) for m,pp in zip(METHODS,formal)]
    status(event='formal_994_aligned',count=994)
    # No profiling work overlaps a formal timed process.
    with ThreadPoolExecutor(max_workers=2) as pool:
        profiles=[pool.submit(job,g,m,0,3,'profile') for g,m in zip((0,4),METHODS)]
        profiles=[p.result() for p in profiles]
    from report import summarize
    summarize(ROOT,merged,profiles)
    (ROOT/'complete.json').write_text(json.dumps({'complete':True,'formal_count':994,'alignment':'passed/full','profiling_count':6}))
    status(event='complete_all',count=994)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if ROOT.exists():status(event='scheduler_failed',error=repr(e))
        raise
