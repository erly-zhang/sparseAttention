"""Submit the reviewed pipeline only after approval bound to its exact inputs."""
import subprocess
from common import HERE,config,require_approval,save


def submit():
    require_approval(allocation=False)
    if (HERE/'submitted.json').exists(): raise PermissionError('Pipeline already submitted')
    c=config()
    jobs={}
    def stage(name,script,dependency=None,array=None):
        command=['sbatch','--parsable']
        if dependency: command+=['--dependency=afterok:'+dependency]
        if array: command+=['--array='+array]
        command+=[str(HERE/script)]
        result=subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit(): raise ValueError('Unexpected sbatch response')
        jobs[name]=job
        save(HERE/'submitted.json',dict(jobs=jobs,stage=name,complete=False))
        return job
    smoke=stage('gpu_smoke','smoke.sbatch')
    fitting=stage('fit','fit.sbatch',smoke,'0-31%32')
    validation=stage('merge_validation','merge-validation.sbatch',fitting)
    dense=stage('dense_replay','replay-dense.sbatch',validation,'0-9%10')
    sparse=stage('sparse_replay','replay-sparse.sbatch',dense,'0-59%60')
    save(HERE/'submitted.json',dict(jobs=jobs,complete=True,
        workflow='GPU smoke -> 32 fit shards -> merge + 551-item validation -> 10 Dense workers -> 60 sparse workers; last completed worker aggregates'))
    print(jobs)


if __name__=='__main__': submit()
