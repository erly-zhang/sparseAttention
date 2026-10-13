"""Submit the user-authorized speed-first suite and its dependent CPU audit."""
import json
import subprocess
from protocol import RUN,sha,save,require_approval
from run import preflight


def submit():
    require_approval('full_suite')
    c,m=preflight()
    v=json.loads((RUN/'cpu-validation.json').read_text())
    assert v['status']=='passed' and v['source_lock_sha256']==sha(RUN/'source-lock.json')
    assert v['input_manifest_sha256']==sha(RUN/'inputs/manifest.json')
    intent=RUN/'submit-intent-full-suite.json'
    if intent.exists():
        raise FileExistsError('submission intent exists; inspect scheduler before any retry')
    stages=('calibration','formal')
    for stage in stages:
        check=subprocess.run(['sbatch','--test-only',str(RUN/('run-'+stage+'.sbatch'))],capture_output=True,text=True)
        assert check.returncode==0,check.stderr
    save(intent,dict(profile='full_suite',config_sha256=sha(RUN/'config.json'),
        source_lock_sha256=sha(RUN/'source-lock.json'),input_manifest_sha256=sha(RUN/'inputs/manifest.json'),
        maximum_gpu_hours=None,formal_workers=len(c['methods'])*c['formal']['shards'],
        resource_priority='time_to_completion',qos='push',blind_automatic_retry=False))
    jobs={}
    previous=None
    for stage in stages:
        command=['sbatch','--parsable']
        if previous:
            command+=['--dependency=afterok:'+jobs[previous],'--kill-on-invalid-dep=yes']
        command.append(str(RUN/('run-'+stage+'.sbatch')))
        result=subprocess.run(command,capture_output=True,text=True)
        for suffix,value in (('stdout',result.stdout),('stderr',result.stderr)):
            with (RUN/('submit-'+stage+'.'+suffix)).open('x') as f:
                f.write(value)
        assert result.returncode==0,result.stderr
        job=result.stdout.strip().split(';')[0]
        assert job.isdigit()
        jobs[stage]=job
        save(RUN/('submitted-'+stage+'.json'),dict(job_id=job,profile=stage,
            dependency=jobs[previous] if previous else None))
        qos=subprocess.run(['scontrol','update','JobId='+job,'QOS=push'],capture_output=True,text=True)
        save(RUN/('qos-update-'+stage+'.json'),dict(returncode=qos.returncode,stdout=qos.stdout,stderr=qos.stderr))
        print('SUBMITTED',stage,job,flush=True)
        previous=stage


if __name__=='__main__':
    submit()
