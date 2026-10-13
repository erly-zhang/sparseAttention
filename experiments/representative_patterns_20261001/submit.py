"""One approved suite: calibration then dependent four-worker array."""
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
    for stage in ('calibration','formal'):
        check=subprocess.run(['sbatch','--test-only',str(RUN/('run-'+stage+'.sbatch'))],capture_output=True,text=True)
        assert check.returncode==0,check.stderr
    save(intent,dict(profile='full_suite',config_sha256=sha(RUN/'config.json'),
        source_lock_sha256=sha(RUN/'source-lock.json'),input_manifest_sha256=sha(RUN/'inputs/manifest.json'),
        maximum_gpu_hours=100,automatic_retry=False))
    jobs={}
    for stage in ('calibration','formal'):
        command=['sbatch','--parsable']
        if stage=='formal':
            command+=['--dependency=afterok:'+jobs['calibration'],'--kill-on-invalid-dep=yes']
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
            dependency=jobs.get('calibration') if stage=='formal' else None))
        print('SUBMITTED',stage,job,flush=True)


if __name__=='__main__':
    submit()
