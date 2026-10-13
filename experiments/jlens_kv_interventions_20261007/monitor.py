"""Persistent pilot supervision with provenance checks and bounded safe retries."""
from pathlib import Path
import json,hashlib,os,subprocess,time
HERE=Path(__file__).resolve().parent
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
TRANSIENT={'NODE_FAIL','PREEMPTED','BOOT_FAIL','TIMEOUT'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def command(args):return subprocess.check_output(args,universal_newlines=True,stderr=subprocess.STDOUT,timeout=60)
def query(job):return [r.split('|') for r in command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode']).splitlines() if r.strip()]
def submit(index):
    job=command(['sbatch','--parsable','--array='+str(index),str(HERE/'pilot.sbatch')]).strip().split(';')[0]
    return job

def reconcile():
    prepared=read(HERE/'prepared.json');completed=[];generated=[];failures=[];source=sha(HERE/'lock.json')
    for trial in prepared['plan']:
        folder=HERE/'outputs'/('trial-%03d'%trial['index'])
        result=folder/'result.json';analysis=folder/'analysis.json'
        if result.exists():
            r=read(result);assert r['lock_sha256']==source and r['trial']==trial
            assert 0<=r['score']<=1 and 0<=r['score_first_50_tokens']<=1
            generated.append({'index':trial['index'],'trial':trial,'score':r['score'],'score_first_50_tokens':r['score_first_50_tokens'],
                              'prediction':r['prediction'],'seconds':r['seconds'],'intervention_relative_norm':r['intervention_relative_norm']})
            if analysis.exists():
                a=read(analysis);assert a['trial']==trial and a['status']=='complete'
                assert a['decomposition_sha256']==sha(folder/'decompositions.pt')
                completed.append(trial['index'])
        if (folder/'failure.json').exists():failures.append({'index':trial['index'],'error':read(folder/'failure.json')})
    save(HERE/'results-summary.json',{'scope':'one fixed full-context case; oracle interventions; trial scores are not task accuracy',
        'trials':105,'generation_completed':len(generated),'analysis_completed':len(completed),'rows':generated,'failures':failures})
    return generated,completed,failures

def main():
    submitted=read(HERE/'submitted.json');baselines=submitted['baseline_job_id'];interventions=submitted['intervention_job_id']
    retries={};start=time.time();last=0
    while time.time()-start<46*3600:
        try:
            generated,completed,failures=reconcile()
            status={'status':'running_or_queued','baseline_job_id':baselines,'intervention_job_id':interventions,
                    'generation_completed':len(generated),'analysis_completed':len(completed),'trials':105,
                    'failures':failures,'retry_jobs':retries,'updated_unix_seconds':time.time()}
            if len(completed)==105:
                status['status']='complete';save(HERE/'latest-status.json',status);print(json.dumps(status),flush=True);return
            scheduler=query(baselines)+query(interventions)
            if any(f['index']<3 for f in failures):
                subprocess.run(['scancel',interventions],check=False)
                status['status']='baseline_validation_failed';save(HERE/'latest-status.json',status);return
            for row in scheduler:
                job,st=row[:2];st=st.split()[0]
                if '_' not in job or not job.rsplit('_',1)[1].isdigit():continue
                index=int(job.rsplit('_',1)[1])
                if index in completed or st not in TRANSIENT:continue
                previous=retries.get(str(index),[])
                if previous:
                    states=[x[1].split()[0] for x in query(previous[-1]) if '_' in x[0]]
                    if not states or any(s not in FAILED|{'COMPLETED'} for s in states):continue
                if len(previous)<2:
                    previous.append(submit(index));retries[str(index)]=previous;save(HERE/'retry-jobs.json',retries)
            active=[row for row in scheduler if row[1].split()[0] not in FAILED|{'COMPLETED'}]
            if not active and not retries and len(completed)<105:
                status['status']='incomplete_needs_inspection';save(HERE/'latest-status.json',status);print(json.dumps(status),flush=True);return
            save(HERE/'latest-status.json',status)
            if time.time()-last>=300:print(json.dumps(status),flush=True);last=time.time()
        except Exception as error:
            save(HERE/'monitor-error.json',{'error':repr(error),'updated_unix_seconds':time.time()})
        time.sleep(60)
    save(HERE/'monitor-expired.json',{'status':'time_limit','retry_jobs':retries})

if __name__=='__main__':main()
