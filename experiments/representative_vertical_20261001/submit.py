"""Exactly one submission per explicitly approved profile; never called by setup."""
import argparse
import json
import subprocess
from protocol import RUN,sha,save,require_approval
from run import preflight,load_heads


def submit(profile):
    require_approval(profile)
    c,m=preflight()
    v=json.loads((RUN/'cpu-validation.json').read_text())
    assert v['status']=='passed' and v['source_lock_sha256']==sha(RUN/'source-lock.json')
    assert v['input_manifest_sha256']==sha(RUN/'inputs/manifest.json')
    if profile=='formal':
        load_heads(c,m)
        audit=json.loads((RUN/'outputs/pilot/independent-audit.json').read_text())
        assert audit['status']=='PASS' and audit['records']==32
    script=RUN/('run-'+profile+'.sbatch')
    # A pre-existing intent blocks retries, including submissions with an unknown result.
    intent=RUN/('submit-intent-'+profile+'.json')
    if intent.exists():
        raise FileExistsError('submission intent exists; inspect scheduler before any retry')
    check=subprocess.run(['sbatch','--test-only',str(script)],capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    save(intent,dict(profile=profile,script_sha256=sha(script),config_sha256=sha(RUN/'config.json'),
        source_lock_sha256=sha(RUN/'source-lock.json'),input_manifest_sha256=sha(RUN/'inputs/manifest.json')))
    result=subprocess.run(['sbatch','--parsable',str(script)],capture_output=True,text=True)
    (RUN/('submit-'+profile+'.stdout')).write_text(result.stdout)
    (RUN/('submit-'+profile+'.stderr')).write_text(result.stderr)
    assert result.returncode==0,result.stderr
    job=result.stdout.strip().split(';')[0]
    assert job.isdigit()
    save(RUN/('submitted-'+profile+'.json'),dict(job_id=job,profile=profile))
    print('SUBMITTED',profile,job)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=('pilot','formal'),default='pilot')
    submit(p.parse_args().profile)
