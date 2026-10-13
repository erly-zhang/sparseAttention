"""Supervise the four missing workers without changing the approved experiment."""
import os
import subprocess
import time
from pathlib import Path

from support import HERE, read, save, sha, verify
from summarize import summarize

LEDGER = HERE / 'recovery-20261009.json'
TERMINAL = {'FAILED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL',
            'CANCELLED', 'OUT_OF_MEMORY', 'COMPLETED'}
TRANSIENT = {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def command(args):
    return subprocess.check_output(args, universal_newlines=True).strip()


def states(job):
    output = command(['sacct', '-j', job, '-X', '-n', '-P',
                      '-o', 'JobID,State'])
    return {row.split('|')[0]: row.split('|')[1].split()[0].split('+')[0]
            for row in output.splitlines() if row.strip()}


def retryable(state, failure, prepared):
    if state in TRANSIENT:
        return True
    if state != 'FAILED' or failure is None:
        return False
    # A metadata assertion may be retried only while local pinned metadata agrees.
    paths = set(prepared['model_files'])
    if failure.get('error') not in {"AssertionError(%r)" % path for path in paths}:
        return False
    for path, expected in prepared['model_files'].items():
        stat = Path(path).stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            return False
    return True


def run():
    ledger = read(LEDGER)
    assert ledger['lock_sha256'] == sha(HERE / 'lock.json')
    assert ledger['recovery_monitor_sha256'] == sha(__file__)
    assert read(HERE / 'model-integrity-20261009.json')['status'] == 'PASS'
    prepared = verify()
    deadline = time.time() + 46 * 3600
    while time.time() < deadline:
        ledger = read(LEDGER)
        summary = summarize()
        detail = {}
        problems = []
        for index in ledger['missing_workers']:
            if index in summary['completed_workers']:
                detail[str(index)] = 'complete'
                continue
            attempts = ledger['attempts'][str(index)]
            current = attempts[-1]
            state = states(current['array_job']).get(current['job_id'], 'UNKNOWN')
            detail[str(index)] = state
            if state not in TERMINAL:
                continue
            failure_path = HERE / ('failure-formal-%d.json' % index)
            failure = read(failure_path) if failure_path.exists() else None
            if len(attempts) >= 3 or not retryable(state, failure, prepared):
                problems.append({'worker': index, 'state': state, 'failure': failure})
                continue
            verify()
            if failure_path.exists():
                archive = HERE / 'recovery-failures-20261009'
                archive.mkdir(exist_ok=True)
                failure_path.replace(archive / ('worker-%d-attempt-%d.json' %
                                               (index, len(attempts))))
            job = command(['sbatch', '--parsable', '--array=' + str(index),
                           str(HERE / 'gpu.sbatch'), 'formal']).split(';')[0]
            attempts.append({'array_job': job, 'job_id': job + '_' + str(index),
                             'submitted_unix': time.time(), 'previous_state': state})
            save(LEDGER, ledger)
            detail[str(index)] = 'resubmitted:' + job
        status = {'status': 'complete' if summary['samples_completed'] == 800 else
                  ('needs_inspection' if problems else 'recovering'),
                  'samples_completed': summary['samples_completed'],
                  'samples_total': 800, 'workers_completed': summary['workers_completed'],
                  'workers_total': 160, 'recovery_workers': detail, 'problems': problems,
                  'updated_unix': time.time(), 'monitor_job': os.environ.get('SLURM_JOB_ID'),
                  'lock_sha256': ledger['lock_sha256']}
        save(HERE / 'latest-status.json', status)
        print(status, flush=True)
        if status['status'] == 'complete':
            return
        if problems:
            raise SystemExit(2)
        time.sleep(300)
    raise SystemExit('Recovery monitor time limit reached; see persisted status.')


if __name__ == '__main__':
    run()
