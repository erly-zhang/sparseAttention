"""Monitor while queued, then hand off atomically to a durable Slurm owner."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from supervisor import HERE, RUN, command, event, now, read, save, verify_control


def bridge():
    save(HERE / 'bridge.json', dict(pid=os.getpid(), started_at_utc=now(), status='running'))
    while not read(HERE / 'state.json').get('complete'):
        with (HERE / 'bridge-gate.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (HERE / 'handoff-request.json').exists():
                event('bridge_handoff', pid=os.getpid())
                save(HERE / 'bridge.json', dict(pid=os.getpid(), status='handed_off', at_utc=now()))
                return
            result = subprocess.run([sys.executable, str(HERE / 'supervisor.py')])
            if result.returncode:
                event('bridge_cycle_failed', return_code=result.returncode)
                if (HERE / 'integrity-error.json').exists():
                    raise RuntimeError('Integrity check failed; dispatch stopped')
        time.sleep(60)


def slurm():
    verify_control(read(HERE / 'control.json'))
    if read(HERE / 'state.json').get('complete'):
        event('standby_not_needed')
        return
    owner = os.environ['SLURM_JOB_ID']
    save(HERE / 'handoff-request.json', dict(job=owner, at_utc=now()))
    # The bridge holds this gate only for a single cycle. Slurm ownership
    # holds it for the full loop, so neither owner can dispatch concurrently.
    with (HERE / 'bridge-gate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        event('slurm_monitor_takeover', job=owner)
        successor_record = HERE / ('successor-' + owner + '.json')
        if not successor_record.exists():
            parent = command(['sbatch', '--parsable', '--dependency=afterany:' + owner,
                              '--comment=jlens-supervisor-successor-of-' + owner,
                              str(HERE / 'monitor.sbatch')]).strip().split(';')[0]
            if not parent.isdigit():
                raise ValueError('Unexpected standby monitor job')
            save(successor_record, dict(job=parent, dependency='afterany:' + owner, at_utc=now()))
            try:
                command(['scontrol', 'update', 'JobId=' + parent, 'QOS=push'])
            except Exception as exc:
                event('standby_qos_update_error', job=parent, error=str(exc))
            event('standby_monitor_submitted', job=parent, parent=owner)
        result = subprocess.run([sys.executable, str(HERE / 'supervisor.py'), '--loop'])
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bridge', action='store_true')
    args = p.parse_args()
    if args.bridge:
        bridge()
    else:
        slurm()
