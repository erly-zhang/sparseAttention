"""Periodic result aggregation and bounded retries of scheduler failures."""
import subprocess
import time
from worker import HERE,read,save,sha

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
TRANSIENT={'TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'}


def summary():
    c=read(HERE/'config.json');p=read(HERE/'prepared.json');rows=[]
    for item in p['evaluation']:
        path=HERE/'outputs'/(item['key']+'.json')
        if not path.exists():continue
        row=read(path)
        assert row['lock_sha256']==sha(HERE/'lock.json') and row['input_ids_sha256']==item['input_ids_sha256']
        rows.append(row)
    tasks={}
    for task in sorted(set(r['task'] for r in p['evaluation'])):
        subset=[r for r in rows if r['task']==task];n=len(subset)
        tasks[task]=dict(completed=n,expected=50,score_percent=100*sum(r['score'] for r in subset)/n if n else None,
            mean_prefill_seconds=sum(r['prefill_seconds'] for r in subset)/n if n else None,
            mean_selector_seconds=sum(r['selector_seconds'] for r in subset)/n if n else None,
            mean_sparse_attention_seconds=sum(r['sparse_attention_seconds'] for r in subset)/n if n else None,
            mean_pair_sparsity=sum(r['pair_sparsity'] for r in subset)/n if n else None)
    result=dict(status='complete' if len(rows)==200 else 'running',completed=len(rows),expected=200,
        tasks=tasks,updated_unix=time.time(),configuration=c,lock_sha256=sha(HERE/'lock.json'))
    controls={}
    for method in ('dense_full','flexprefill','minference'):
        controls[method]={}
        for task in tasks:
            refs=[r for r in p['paired_historical_baselines'] if r['method']==method and r['task']==task]
            assert len(refs)==50
            controls[method][task]=dict(samples=50,score_percent=100*sum(r['score'] for r in refs)/50,
                mean_prefill_seconds=sum(r['prefill_seconds'] for r in refs)/50)
    result['historical_baselines']=controls
    result['timing_note']='Historical baselines used their original instrumentation; MASA kernel JIT is warmed before timing. Task inputs and generation budgets are identical.'
    save(HERE/'results-summary.json',result)
    return result


def query(ids):
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobID,State,ExitCode'],universal_newlines=True,timeout=45)
    return [line.split('|') for line in text.splitlines() if line.strip()]


def main():
    ledger=read(HERE/'submitted.json');retries=read(HERE/'retries.json') if (HERE/'retries.json').exists() else {}
    started=time.time()
    while time.time()-started < 46*3600:
        try:
            result=summary();jobs=query([ledger['proof'],ledger['formal']]+[j for v in retries.values() for j in v])
            result['jobs']=jobs;result['retries']=retries
            for job,state,exitcode,*_ in jobs:
                state=state.split()[0]
                if state not in TRANSIENT or not job.startswith(ledger['formal']+'_'):continue
                tail=job.rsplit('_',1)[-1]
                if not tail.isdigit():continue
                index=int(tail);key=str(index)
                if (HERE/'completed'/(key+'.json')).exists():continue
                previous=retries.get(key,[])
                if previous:
                    last=[r for r in jobs if r[0].split('_')[0]==previous[-1]]
                    if not last or any(r[1].split()[0] not in TRANSIENT for r in last):continue
                if len(previous)>=2:continue
                retry=subprocess.check_output(['sbatch','--parsable','--array='+key,str(HERE/'gpu.sbatch'),'formal'],
                    universal_newlines=True,timeout=45).strip().split(';')[0]
                retries.setdefault(key,[]).append(retry);save(HERE/'retries.json',retries)
            result['failures']=[str(f) for f in (HERE/'failures').glob('*.json')]
            save(HERE/'latest-status.json',result)
            print('PROGRESS',result['completed'],result['tasks'],'failures',result['failures'],flush=True)
            if result['completed']==200:return
            live=any(r[1].split()[0] not in TERMINAL for r in jobs)
            if jobs and not live:
                result['status']='needs_inspection';save(HERE/'latest-status.json',result);return
        except Exception as error:save(HERE/'monitor-error.json',dict(error=repr(error),unix=time.time()))
        time.sleep(60)


if __name__=='__main__':main()
