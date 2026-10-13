"""Persistent controller, owned-job retries, verification and final publication."""
import subprocess,sys,time,json,os,fcntl
from pathlib import Path
from common import HERE,read,save,verify

def command(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout
def submit(indices):
    job=command(['sbatch','--parsable','--array='+','.join(map(str,indices)),str(HERE/'worker.sbatch')]).strip().split(';')[0]
    state=read(HERE/'controller.json');state['arrays'].append(job)
    for i in indices:state['last_jobs'][str(i)]=job+'_'+str(i)
    save(HERE/'controller.json',state)
    command(['scontrol','update','JobId='+job,'QOS=push'])
    print('SUBMITTED',job,indices,flush=True)

def main():
    with (HERE/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify();log=(HERE/'source-verification.log').open('a')
        checker=subprocess.Popen([sys.executable,'-u','verify_sources.py'],cwd=str(HERE),stdout=log,stderr=log)
        while True:
            state=read(HERE/'controller.json');done=[i for i in range(128) if (HERE/'outputs'/('worker-'+str(i))/'complete.json').exists()]
            save(HERE/'progress.json',dict(completed_workers=len(done),total_workers=128,completed_method_sample_layers=len(done)*350,
                total_method_sample_layers=44800,source_verification_complete=(HERE/'source-verification.json').exists(),
                monitor_job=os.environ.get('SLURM_JOB_ID'),updated_epoch=time.time(),phase='decomposition',blocked=state.get('blocked',{})))
            print('PROGRESS',len(done),'/128',flush=True)
            if checker.poll() is not None and checker.returncode and not (HERE/'source-verification.json').exists():raise RuntimeError('Source verification failed')
            if len(done)==128 and (HERE/'source-verification.json').exists():
                subprocess.run([sys.executable,'-u','aggregate.py'],check=True,cwd=str(HERE))
                save(HERE/'progress.json',dict(phase='complete',completed_workers=128,total_workers=128,audit='PASS',completed_epoch=time.time()))
                return
            rows=command(['sacct','-X','-n','-P','-j',','.join(state['arrays']),'--format=JobID,State']).splitlines()
            observed={r.split('|')[0]:r.split('|')[1].split()[0] for r in rows if '|' in r}
            retries=[]
            for i in range(128):
                if i in done or str(i) in state.get('blocked',{}):continue
                job=state['last_jobs'][str(i)];status=observed.get(job,'PENDING')
                if status in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','CANCELLED']:
                    p=HERE/('slurm-'+job+'.out');text=p.read_text(errors='replace')[-6000:] if p.exists() else ''
                    if status in ['TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'] or any(t in text for t in ['out of memory','OutOfMemoryError','Insufficient free GPU memory','Input/output error']):retries.append(i)
                    else:state.setdefault('blocked',{})[str(i)]=dict(job=job,status=status,log=str(p),tail=text)
            save(HERE/'controller.json',state)
            if retries:submit(retries)
            time.sleep(60)
if __name__=='__main__':main()
