"""Submit only after a separately recorded, source-bound user approval."""
import subprocess
from prepare import HERE, read, sha, save


def main():
    approval = read(HERE/'approval.json')
    assert approval.get('approved') is True
    assert approval['lock_sha256'] == sha(HERE/'lock.json')
    for p,h in read(HERE/'lock.json')['files'].items():
        assert sha(p) == h, p
    if (HERE/'submitted.json').exists():
        raise ValueError('Already submitted; inspect progress before retrying')
    count = len(read(HERE/'prepared.json')['plan'])
    result = subprocess.run(['sbatch','--parsable','--array=0-2',str(HERE/'pilot.sbatch')],
                            capture_output=True,text=True,check=True)
    baseline_id = result.stdout.strip().split(';')[0]
    subprocess.run(['scontrol','update','JobId='+baseline_id,'QOS=push'],
                   capture_output=True,text=True,check=True)
    submitted = dict(baseline_job_id=baseline_id,trials=count,
                     lock_sha256=sha(HERE/'lock.json'),concurrency_cap=None)
    save(HERE/'submitted.json',submitted)
    # Full-context generation/analysis parity must pass before interventions.
    result = subprocess.run(['sbatch','--parsable',f'--dependency=afterok:{baseline_id}',
                             '--kill-on-invalid-dep=yes',f'--array=3-{count-1}',str(HERE/'pilot.sbatch')],
                            capture_output=True,text=True,check=True)
    submitted['intervention_job_id'] = result.stdout.strip().split(';')[0]
    subprocess.run(['scontrol','update','JobId='+submitted['intervention_job_id'],'QOS=push'],
                   capture_output=True,text=True,check=True)
    save(HERE/'submitted.json',submitted)
    print(submitted)


if __name__ == '__main__':
    main()
