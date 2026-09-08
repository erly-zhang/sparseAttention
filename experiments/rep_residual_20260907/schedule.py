"""Isolated calibration, smoke and aligned full evaluation scheduler."""
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

HERE=Path(__file__).resolve().parent
ROOT=Path('/local/results/rep_residual_20260907')
WORK=Path('/home/ubuntu/work')
PY='/home/ubuntu/miniconda3/envs/attentionmap/bin/python'
REFERENCE=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl'
GROUP=WORK/'experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json'
METHODS=['probmean_shared_residual_compact10240','probmean_shared_residual_topp99_matched']
SHARDS=[(0,125),(125,250),(250,375),(375,497)]


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def identity(r):return (r['input_ids_sha256'],r['input_tokens'])
def status(**entry):
    with (ROOT/'scheduler_status.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**entry})+'\n')


def validate(path,expected,calibration=False):
    summary=json.loads((path/'summary.json').read_text())
    records=rows(path/'online_metrics.jsonl');ref=rows(REFERENCE)
    assert len(records)==expected and len({identity(x) for x in records})==expected
    assert summary['run_args']['max_new_tokens']==128
    if calibration:
        assert not {identity(x) for x in records}&{identity(x) for x in ref}
    else:
        got=[identity(x) for x in records]
        assert [identity(x) for x in ref if identity(x) in set(got)]==got
        assert summary['input_alignment']['status']=='passed'
        assert summary['method']==path.parent.name
        assert summary['method_metadata']['num_groups_per_layer']==3
        assert summary['method_metadata']['online_config']['policy_sha256']==hashlib.sha256((ROOT/'policies.json').read_bytes()).hexdigest()
    samples=json.loads((path/'lm_eval_results.json').read_text())['samples']['kv_retrieval']
    assert len(samples)==expected
    score=sum(x['get_score_one_kv_retrieval'] for x in samples)
    assert abs(score/expected-summary['official_lm_eval_results']['kv_retrieval']['get_score_one_kv_retrieval,none'])<1e-9
    return summary,records,samples


def job(gpu,method,a,b,phase):
    calibration=phase=='calibration'
    base=ROOT/'attempts'/phase/f'{method}_{a}_{b}'
    base.mkdir(parents=True,exist_ok=True)
    for attempt in sorted(base.glob('attempt*')):
        path=attempt/method/'kv_retrieval'
        try:
            validate(path,1 if calibration else b-a,calibration)
            if calibration:assert len(list((attempt/'layers').glob('layer*.json')))==32
            return path
        except (FileNotFoundError,AssertionError,KeyError,json.JSONDecodeError):pass
    lock=(ROOT/f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuids[gpu] in active:raise RuntimeError(f'GPU {gpu} occupied; no existing process was stopped')
    attempt=base/f'attempt{len(list(base.glob("attempt*")))+1}';attempt.mkdir()
    folder=f'calibration_{a}' if calibration else ('smoke_0_3' if phase=='smoke' else f'formal_{a}_{b}')
    cmd=[PY,str(HERE/('calibrate.py' if calibration else 'online.py')),
         '--model',str(WORK/'model/Llama-3.1-8B-Instruct'),'--method',method,'--task','kv_retrieval',
         '--max_length','131072','--batch_size','1','--seed','42','--chat',
         '--group_config',str(GROUP),'--flexprefill_root',str(WORK/'FlexPrefill'),
         '--task_config_dir',str(ROOT/'data'/folder/'task_configs'),'--record_sparsity',
         '--fixed_topk_budget','10240','--output_dir',str(attempt)]
    cmd+=['--create_reference'] if calibration else ['--reference_metrics',str(REFERENCE)]
    env={**os.environ,'PYTHONPATH':str(WORK),'CUDA_VISIBLE_DEVICES':str(gpu),
         'TOKENIZERS_PARALLELISM':'false','CALIBRATION_OUTPUT':str(attempt/'layers'),
         'RESIDUAL_POLICY':str(ROOT/'policies.json')}
    (attempt/'command.json').write_text(json.dumps(cmd,indent=2))
    with (attempt/'run.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(event='start',pid=process.pid,gpu=gpu,method=method,phase=phase,a=a,b=b,attempt=str(attempt))
        code=process.wait()
    if code:raise RuntimeError(f'{phase} {method} failed: {attempt}/run.log code={code}')
    path=attempt/method/'kv_retrieval'
    validate(path,1 if calibration else b-a,calibration)
    status(event='complete',method=method,phase=phase,a=a,b=b)
    return path


def merge(method,paths):
    dest=ROOT/'merged'/method/'kv_retrieval'
    if (dest/'summary.json').exists():
        summary=json.loads((dest/'summary.json').read_text())
        assert summary['input_alignment']['count']==497
        return
    if dest.exists():raise RuntimeError(f'Refusing to overwrite partial merge {dest}')
    results=[validate(p,b-a) for p,(a,b) in zip(paths,SHARDS)]
    rr=[x for _,records,_ in results for x in records]
    predictions=[x for _,_,samples in results for x in samples]
    reference=rows(REFERENCE)
    assert [identity(x) for x in rr]==[identity(x) for x in reference]
    assert len(rr)==len({identity(x) for x in rr})==497
    assert len(predictions)==len({x['doc_hash'] for x in predictions})==497
    correct=sum(x['get_score_one_kv_retrieval'] for x in predictions)
    for i,row in enumerate(rr):row['call_index']=i
    latency=np.array([x['prefill_latency_sec'] for x in rr])
    selected=sum(x['selected_token_pairs'] for x in rr);causal=sum(x['causal_token_pairs'] for x in rr)
    assert causal==sum(32*32*x['input_tokens']*(x['input_tokens']+1)//2 for x in rr)
    runtime={'count':497,'correct':correct,'accuracy':correct/497,
        'avg_prefill_latency_sec':float(latency.mean()),'median_prefill_latency_sec':float(np.median(latency)),
        'p90_prefill_latency_sec':float(np.quantile(latency,.9)),
        'avg_decode_latency_sec':float(np.mean([x['decode_latency_sec'] for x in rr])),
        'avg_total_latency_sec':float(np.mean([x['total_latency_sec'] for x in rr])),
        'selected_token_pairs':selected,'causal_token_pairs':causal,'global_token_sparsity':1-selected/causal,
        'max_peak_memory_bytes':max(x.get('peak_memory_bytes',0) for x in rr)}
    result={**results[0][0],'runtime':runtime,'source_shards':[str(p) for p in paths],
        'input_alignment':{'status':'passed','alignment_scope':'full','count':497,'reference_count':497,
                           'reference_metrics':str(REFERENCE),'compared_fields':['input_tokens','input_ids_sha256']},
        'official_lm_eval_results':{'kv_retrieval':{'get_score_one_kv_retrieval,none':correct/497}}}
    dest.mkdir(parents=True)
    (dest/'online_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    status(event='merged',method=method,correct=correct,count=497)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['calibration','fixed','both'],default='calibration')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'correctness.json').read_text())['passed']
    assert json.loads((ROOT/'data_manifest.json').read_text())['calibration_evaluation_disjoint']
    status(event='scheduler_started',pid=os.getpid(),stage=args.stage)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(job,i,'probmean_perhead_compact10240',i,i+1,'calibration') for i in range(3)]
        completed=[f.result() for f in fs]
    for i,path in enumerate(completed):
        target=ROOT/'calibration'/str(i);target.parent.mkdir(exist_ok=True)
        source=path.parent.parent/'layers'
        assert len(list(source.glob('layer*.json')))==32
        if not target.exists():target.symlink_to(source,target_is_directory=True)
    subprocess.run([PY,str(HERE/'fit.py')],check=True)
    if args.stage=='calibration':
        status(event='calibration_complete');return
    if args.stage=='both':
        assert (ROOT/'topp_matched_confirmed.json').exists(), 'Top-p residual interpretation awaits user choice'
    methods=METHODS if args.stage=='both' else METHODS[:1]
    with ThreadPoolExecutor(max_workers=len(methods)) as pool:
        smoke=[pool.submit(job,i,m,0,3,'smoke') for i,m in enumerate(methods)]
        smoke_paths=[f.result() for f in smoke]
    for path in smoke_paths:
        _,records,_=validate(path,3)
        assert all(x['max_probability_sum_error']<1e-4 for x in records)
    frozen=ROOT/'formal_source';frozen.mkdir(exist_ok=True)
    for source in HERE.glob('*.py'):
        target=frozen/source.name
        if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('Source changed after freezing')
        if not target.exists():shutil.copy2(source,target)
    with ThreadPoolExecutor(max_workers=4*len(methods)) as pool:
        futures={m:[pool.submit(job,mi*4+si,m,a,b,'formal') for si,(a,b) in enumerate(SHARDS)] for mi,m in enumerate(methods)}
        for m,fs in futures.items():merge(m,[f.result() for f in fs])
    from report import summarize
    summarize(ROOT)
    status(event='requested_stage_complete',stage=args.stage,count=497*len(methods))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if ROOT.exists():status(event='scheduler_failed',error=repr(error))
        raise
