"""CPU supervision, source-bound summaries and bounded transient retries."""
import json
import subprocess
import time
from support import HERE,read,save
from summarize import summarize

TRANSIENT={'TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
FAILED=TRANSIENT|{'FAILED','OUT_OF_MEMORY','CANCELLED','DEADLINE'}


def query(jobs):
    output=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),
        '--format=JobID,State,ExitCode'],universal_newlines=True,timeout=60)
    return [line.split('|') for line in output.splitlines() if line.strip()]


def retry_candidates(original_rows,retry_rows,retries,completed):
    """Only scheduler-classified transient failures; at most two retries per worker."""
    candidates=[];exhausted=[]
    for row in original_rows:
        job,state=row[:2];state=state.split()[0]
        tail=job.rsplit('_',1)[-1]
        if '_' not in job or not tail.isdigit():continue
        index=int(tail)
        if index in completed or state not in TRANSIENT:continue
        previous=retries.get(str(index),[])
        if previous:
            last=previous[-1]
            states=[r[1].split()[0] for r in retry_rows if r[0].split('_')[0]==last]
            if not states or any(s not in FAILED|{'COMPLETED'} for s in states):continue
            if any(s not in TRANSIENT for s in states):exhausted.append(index);continue
        if len(previous)<2:candidates.append(index)
        else:exhausted.append(index)
    return candidates,exhausted


def main():
    ledger=read(HERE/'submitted.json');started=time.time()
    retries=read(HERE/'retry-jobs.json') if (HERE/'retry-jobs.json').exists() else {}
    while time.time()-started < 46*3600:
        try:
            status=summarize();status['updated_unix_seconds']=time.time();status['jobs']=ledger
            stage_rows=query([ledger[k] for k in ['proof','history']])
            formal_rows=query([ledger['formal']])
            retry_ids=[job for jobs in retries.values() for job in jobs]
            retry_rows=query(retry_ids) if retry_ids else []
            candidates,exhausted=retry_candidates(formal_rows,retry_rows,retries,set(status['completed_workers']))
            for index in candidates:
                job=subprocess.check_output(['sbatch','--parsable','--array='+str(index),str(HERE/'gpu.sbatch'),'formal'],
                    universal_newlines=True,timeout=60).strip().split(';')[0]
                retries.setdefault(str(index),[]).append(job)
                save(HERE/'retry-jobs.json',retries)
            status['retry_jobs']=retries;status['retry_exhausted_workers']=exhausted
            status['failure_files']=[str(p) for p in HERE.glob('failure-*.json')]
            status['scheduler_failed']=['|'.join(row) for row in stage_rows+formal_rows+retry_rows
                if row[1].split()[0] in FAILED]
            save(HERE/'latest-status.json',status)
            print(json.dumps({k:status[k] for k in ['status','samples_completed','workers_completed','failure_files','scheduler_failed','retry_jobs']}),flush=True)
            if status['status']=='complete':return
            terminal_stage=any(r[1].split()[0] in FAILED for r in stage_rows)
            terminal_formal=any(r[1].split()[0] in FAILED-TRANSIENT for r in formal_rows+retry_rows)
            if status['failure_files'] or terminal_stage or terminal_formal or exhausted:
                status['status']='needs_inspection';save(HERE/'latest-status.json',status);raise SystemExit(2)
        except Exception as error:
            save(HERE/'monitor-error.json',dict(error=repr(error),updated_unix_seconds=time.time()))
        time.sleep(300)

if __name__=='__main__':main()
