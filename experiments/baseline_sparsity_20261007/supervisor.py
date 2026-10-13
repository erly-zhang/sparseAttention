"""Persistent scheduler supervision; never touches unrelated user allocations."""
import json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def command(args):return subprocess.check_output(args,universal_newlines=True,stderr=subprocess.STDOUT,timeout=60)
def query(job):
    return [line.split('|') for line in command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode']).splitlines() if line.strip()]
def submit(script,index=None):
    cmd=['sbatch','--parsable']
    if index is not None:cmd+=['--array='+str(index)]
    job=command(cmd+[str(HERE/script)]).strip().split(';')[0]
    command(['scontrol','update','JobId='+job,'QOS=push'])
    return job

def main():
    state=json.loads((HERE/'submitted.json').read_text());original_workers=state['worker_job'];smoke=state['smoke_job']
    retry_jobs={};start=time.time();last=0
    while time.time()-start<46*3600:
        done=(HERE/'report-update.json').exists()
        completed=[];failures=[]
        for p in (HERE/'outputs').glob('worker-*/complete.json'):completed.append(p.parent.name)
        for p in (HERE/'outputs').glob('*/failure.json'):failures.append({'worker':p.parent.name,'failure':json.loads(p.read_text())})
        status={'status':'complete_pdf_generated' if done else 'running_or_queued','smoke_job':smoke,'worker_job':original_workers,
            'completed_workers':len(completed),'total_workers':80,'failures':failures,'retry_jobs':retry_jobs,
            'observed_records':sum(1 for p in (HERE/'outputs').glob('worker-*/*.json') if p.name not in ['complete.json','start.json','progress.json','failure.json']),
            'updated_unix_seconds':time.time()}
        save(HERE/'latest-status.json',status)
        if done:return
        try:
            sm=query(smoke)
            smoke_fail=[x for x in sm if x[1].split()[0] in FAILED]
            if smoke_fail:
                # Remove only this experiment's dependent array, so it cannot wait forever.
                subprocess.run(['scancel',original_workers],check=False)
                status['status']='smoke_failed';status['scheduler_failures']=smoke_fail;save(HERE/'latest-status.json',status)
                return
            if all((HERE/'outputs'/('smoke-'+m)/'complete.json').exists() for m in ['flexprefill','minference']):
                for fields in query(original_workers):
                    jid,st=fields[:2];st=st.split()[0]
                    if '_' not in jid or not jid.rsplit('_',1)[1].isdigit():continue
                    index=int(jid.rsplit('_',1)[1]);folder=HERE/'outputs'/('worker-'+str(index))
                    if (folder/'complete.json').exists() or st not in FAILED:continue
                    # Retry transient scheduler/node/time failures. Code failures need inspection.
                    if st not in {'NODE_FAIL','PREEMPTED','BOOT_FAIL','TIMEOUT'}:continue
                    previous=retry_jobs.get(str(index),[])
                    if previous:
                        latest=query(previous[-1]);states=[r[1].split()[0] for r in latest if '_' in r[0]]
                        if not states or any(s not in FAILED|{'COMPLETED'} for s in states):continue
                    if len(previous)<2:
                        previous.append(submit('worker.sbatch',index));retry_jobs[str(index)]=previous
                        save(HERE/'retry-jobs.json',retry_jobs)
                if len(completed)==80:
                    # Retry report-only failures without replaying completed samples.
                    source=HERE.parent/'jlens_prefill_analysis_20261005/env.sh'
                    cmd='source '+str(source)+'\n"$BASE/env/bin/python" '+str(HERE/'aggregate.py')
                    proc=subprocess.run(['/bin/bash','-c',cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=1800)
                    (HERE/'supervisor-finalize.log').write_text(proc.stdout)
                    if proc.returncode:status['status']='report_update_failed';save(HERE/'latest-status.json',status);return
        except Exception as error:
            status['supervisor_error']=str(error);save(HERE/'latest-status.json',status)
        if time.time()-last>=300:print(json.dumps(status),flush=True);last=time.time()
        time.sleep(60)
    save(HERE/'supervisor-expired.json',{'status':'time_limit','retry_jobs':retry_jobs})

if __name__=='__main__':main()
