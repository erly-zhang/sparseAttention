"""Persistent Slurm supervision; identical scientific code, per-input scheduling."""
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
TRANSIENT = {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'OUT_OF_MEMORY'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode:
        raise RuntimeError('Command failed: {}: {}'.format(args[0], p.stderr.strip()))
    return p.stdout


def event(name, **fields):
    row = dict(at_utc=now(), event=name, **fields)
    with (HERE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def identity(method, index):
    return method + ':' + str(index)


def result_path(method, item):
    return RUN / 'outputs' / ('replay-' + method + '-' + str(item['analysis_shard'])) / (item['key'] + '.json')


def retryable(state, text):
    return state in TRANSIENT or any(s in text for s in [
        'CUDA out of memory', 'OutOfMemoryError', 'Insufficient free allocated GPU memory',
        'CUDA error: an illegal memory access', 'CUDA error: unspecified launch failure',
        'Failed to initialize NVML', 'Connection timed out', 'Input/output error'])


def ready(method, index, tasks):
    task = tasks[identity(method, index)]
    return (not task['done'] and not task.get('blocked') and
            (method == 'dense_full' or tasks[identity('dense_full', index)]['done']))


def scheduler(roots):
    if not roots:
        return {}
    rows = command(['sacct', '-X', '-n', '-P', '-j', ','.join(roots),
                    '--format=JobID,State,ExitCode,NodeList'])
    states = {}
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) >= 4 and fields[0]:
            states[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'),
                                   exit_code=fields[2], node=fields[3])
    # Queue state takes precedence over possibly delayed accounting rows.
    # Completed jobs may have expired from squeue while still in sacct.
    queue = command(['squeue', '-h', '-r', '-u', 'jzhang114', '-o', '%i|%T|%N|%R'])
    for line in queue.splitlines():
        f = line.split('|')
        if len(f) >= 4 and f[0].split('_')[0] in roots:
            states[f[0]] = dict(state=f[1], node=f[2], reason=f[3], exit_code=None)
    return states


def verify_control(control):
    for path, expected in control['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Pinned control or scientific source changed: ' + path)
    approval = read(RUN / 'approval.json')
    if not approval.get('approved'):
        raise PermissionError('Experiment approval revoked')


def validate_result(path, method, item, control):
    row = read(path)
    if row['method'] != method or row['key'] != item['key']:
        raise ValueError('Wrong result identity')
    if row['positions'] != item['analysis_positions']:
        raise ValueError('Wrong observed positions')
    if row['lens_sha256'] != control['lens_sha256'] or row['config_sha256'] != control['config_sha256']:
        raise ValueError('Mixed lens or scientific configuration')
    if set(map(int, row['layer_metrics'])) != set(range(32)):
        raise ValueError('Missing layer observations')
    if not (path.parent / row['artifact_file']).exists():
        raise ValueError('Missing saved activation artifact')
    for metrics in row['layer_metrics'].values():
        if any(not math.isfinite(x) for x in metrics['js_nats']):
            raise ValueError('Nonfinite J-lens divergence')


def submit(method, indices, state):
    # One array index is one original input; no cap on total GPUs or attempts.
    args = ['sbatch', '--parsable', '--array=' + ','.join(map(str, indices)),
            '--exclude=' + ','.join(sorted(state['excluded_nodes'])),
            '--export=ALL,JLENS_METHOD=' + method, str(HERE / 'replay-one.sbatch')]
    parent = command(args).strip().split(';')[0]
    if not parent.isdigit():
        raise ValueError('Unexpected submission response')
    # Persist ownership before further scheduler changes to prevent duplicates.
    state['roots'].append(parent)
    for index in indices:
        state['tasks'][identity(method, index)]['attempts'].append(dict(
            parent=parent, job=parent + '_' + str(index), at_utc=now()))
    save(HERE / 'state.json', state)
    try:
        command(['scontrol', 'update', 'JobId=' + parent, 'QOS=push'])
    except Exception as exc:
        event('qos_update_error', job=parent, error=str(exc))
    event('submitted', method=method, indices=indices, parent=parent)


def cycle(control, state, items, methods):
    verify_control(control)
    states = scheduler(state['roots'])
    for method in methods:
        for index, item in enumerate(items):
            task = state['tasks'][identity(method, index)]
            if task['done']:
                continue
            path = result_path(method, item)
            if path.exists():
                validate_result(path, method, item, control)
                task['done'] = True
                task['completed_at_utc'] = now()
                event('result_saved', method=method, key=item['key'], index=index)
                continue
            if not task['attempts']:
                continue
            attempt = task['attempts'][-1]
            observed = states.get(attempt['job'], states.get(attempt['parent'], {}))
            task['scheduler'] = observed
            current = observed.get('state', 'UNKNOWN')
            if current in ACTIVE or current == 'UNKNOWN' or attempt.get('handled'):
                continue
            log = HERE / ('slurm-sample-' + attempt['parent'] + '_' + str(index) + '.out')
            text = log.read_text(errors='replace')[-16000:] if log.exists() else ''
            attempt['handled'] = True
            if retryable(current, text):
                if ('out of memory' in text or 'OutOfMemoryError' in text or
                        'Insufficient free allocated GPU memory' in text):
                    node = observed.get('node', '')
                    if node.startswith('w') and '[' not in node:
                        state['excluded_nodes'] = sorted(set(state['excluded_nodes'] + [node]))
                event('retry_transient', method=method, index=index, job=attempt['job'],
                      scheduler_state=current, node=observed.get('node'))
            else:
                # Do not repeatedly rerun a mathematical or parsing error.
                task['blocked'] = dict(job=attempt['job'], state=current, log=str(log), tail=text[-3000:])
                event('scientific_or_unknown_error', method=method, index=index, job=attempt['job'], log=str(log))
    save(HERE / 'state.json', state)
    for method in methods:
        indices = []
        for index in range(len(items)):
            task = state['tasks'][identity(method, index)]
            if ready(method, index, state['tasks']) and (not task['attempts'] or task['attempts'][-1].get('handled')):
                indices.append(index)
        if indices:
            try:
                submit(method, indices, state)
            except Exception as exc:
                event('submission_error', method=method, error=str(exc))
    counts = {m: sum(state['tasks'][identity(m, i)]['done'] for i in range(len(items))) for m in methods}
    alerts = {k: t['blocked'] for k, t in state['tasks'].items() if t.get('blocked')}
    save(HERE / 'alerts.json', alerts)
    old = read(RUN / 'run-status.json')
    old.update(updated_at_utc=now(), phase='InfiniteBench per-sample Dense and sparse analysis',
               current_gpu_validation_retry_complete=True, validation_completed_items=551,
               merged_lens_complete=True, completed_fit_shards=32, completed_fit_prompts=1000,
               evaluation_completed_prefills=sum(counts.values()), evaluation_total_prefills=len(items) * len(methods),
               evaluation_completed_by_method=counts, supervisor_job=os.environ.get('SLURM_JOB_ID'),
               supervisor_interval_seconds=60, supervisor_heartbeat_utc=now(),
               sample_array_jobs=state['roots'], automatic_retries=True,
               automatic_retry_gpu_budget=None, excluded_nodes=state['excluded_nodes'],
               blocked_sample_jobs=len(alerts), formal_analysis_complete=False,
               scheduler_states=dict(Counter(v.get('state', 'UNKNOWN') for v in states.values())))
    save(RUN / 'run-status.json', old)
    state['last_heartbeat_utc'] = now()
    save(HERE / 'state.json', state)
    event('heartbeat', completed=sum(counts.values()), total=len(items) * len(methods), by_method=counts, alerts=len(alerts))
    if sum(counts.values()) == len(items) * len(methods):
        # The original aggregator verifies every artifact, pairing and score label.
        command(['bash', str(HERE / 'aggregate.sh')])
        audit = read(RUN / 'outputs/independent-audit.json')
        if audit['status'] != 'PASS' or audit['records'] != 1400:
            raise ValueError('Final audit failed')
        old.update(phase='Complete', formal_analysis_complete=True, completed_at_utc=now())
        save(RUN / 'run-status.json', old)
        state['complete'] = True
        save(HERE / 'state.json', state)
        event('experiment_complete', records=1400)
        return True
    return False


def main(loop=False):
    with (HERE / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = read(HERE / 'control.json')
        verify_control(control)
        manifest = read(RUN / 'input-manifest.json')
        methods = read(RUN / 'config.json')['replay']['methods']
        state = read(HERE / 'state.json')
        while not state.get('complete'):
            try:
                if cycle(control, state, manifest['evaluation'], methods):
                    break
            except (ValueError, PermissionError) as exc:
                event('integrity_error', error=str(exc))
                save(HERE / 'integrity-error.json', dict(at_utc=now(), error=str(exc)))
                raise
            except Exception as exc:
                event('monitor_transient_error', error=repr(exc))
            if not loop:
                break
            time.sleep(60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--loop', action='store_true')
    args = p.parse_args()
    main(args.loop)
