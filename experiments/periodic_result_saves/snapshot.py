"""Save immutable, deduplicated experiment snapshots without modifying workers."""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def store_blob(destination, data):
    digest = hashlib.sha256(data).hexdigest()
    path = destination / 'objects' / digest[:2] / digest
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(digest + '.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    if path.stat().st_size != len(data):
        raise ValueError('Invalid saved object size: ' + str(path))
    return {'sha256': digest, 'bytes': len(data),
            'object': str(path.relative_to(destination))}


def snapshot(run, destination):
    """Only count sample records whose matching logits are complete and verified."""
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination == run or run / 'outputs' in destination.parents:
        raise ValueError('Snapshot destination must be outside experiment outputs')
    files, warnings, scores = {}, [], collections.defaultdict(list)
    completed, failures, observed_tasks = [], [], collections.Counter()
    paths = list(run.glob('*.json')) + [run / 'inputs' / 'manifest.json']
    paths += list((run / 'outputs').rglob('*.json'))
    for path in sorted(paths):
        if not path.exists():
            continue
        relative = str(path.relative_to(run))
        try:
            data = path.read_bytes()
            value = json.loads(data)
            is_sample = (path.parent.name.startswith('formal-') and
                         isinstance(value, dict) and 'prediction' in value)
            is_observation = (path.parent.name.startswith('worker-') and isinstance(value, dict)
                              and 'layers' in value and 'passes' in value and 'key' in value)
            if is_sample:
                logits = path.parent / value['logits_file']
                if logits.parent.resolve() != path.parent.resolve():
                    raise ValueError('Invalid logits path')
                logits_data = logits.read_bytes()
                if hashlib.sha256(logits_data).hexdigest() != value['logits_sha256']:
                    raise ValueError('Logits not complete or hash mismatch')
                files[str(logits.relative_to(run))] = store_blob(destination, logits_data)
            if is_observation:
                for observed in value['passes']:
                    logits = path.parent / observed['logits_file']
                    if logits.parent.resolve() != path.parent.resolve():
                        raise ValueError('Invalid diagnostic logits path')
                    logits_data = logits.read_bytes()
                    if hashlib.sha256(logits_data).hexdigest() != observed['logits_sha256']:
                        raise ValueError('Diagnostic logits not complete or hash mismatch')
                    files[str(logits.relative_to(run))] = store_blob(destination, logits_data)
            files[relative] = store_blob(destination, data)
            if is_sample:
                scores[(value['method'], value['task'])].append(float(value['score']))
            if is_observation:
                observed_tasks[value['task']] += 1
            if path.name == 'complete.json' and path.parent.name.startswith('formal-'):
                completed.append(path.parent.name)
            if path.name == 'failure.json':
                failures.append(relative)
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append({'file': relative, 'reason': str(error)})
    counts = collections.Counter()
    task_scores = {}
    for (method, task), values in sorted(scores.items()):
        counts[method] += len(values)
        task_scores.setdefault(method, {})[task] = {
            'completed': len(values), 'mean_score_so_far': sum(values) / len(values)}
    stamp = timestamp()
    value = {'saved_at_utc': stamp, 'run': str(run),
             'formal_records': sum(counts.values()), 'by_method': dict(counts),
             'observed_records': sum(observed_tasks.values()), 'observations_by_task': dict(observed_tasks),
             'task_scores': task_scores, 'scores_are_provisional': True,
             'completed_formal_workers': completed, 'failure_files': failures,
             'skipped_files': warnings, 'files': files,
             'final_audit_saved': 'outputs/independent-audit.json' in files}
    filename = stamp.replace(':', '').replace('+', '_') + '-' + uuid.uuid4().hex[:8] + '.json'
    write_json(destination / 'snapshots' / filename, value)
    write_json(destination / 'latest.json', value)
    return value, filename


def serve(run, destination, interval):
    if interval <= 0:
        raise ValueError('interval must be positive')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'pid': os.getpid(), 'host': socket.gethostname(),
                 'interval_seconds': interval, 'started_at_utc': timestamp(),
                 'run': str(run.resolve()), 'status': 'running',
                 'snapshotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        while True:
            start = time.monotonic()
            try:
                saved, filename = snapshot(run, destination)
                state.update(last_saved_at_utc=saved['saved_at_utc'],
                             last_snapshot=filename, formal_records=saved['formal_records'],
                             observed_records=saved['observed_records'],
                             skipped_files=saved['skipped_files'], last_error=None)
                print('{} saved {} formal and {} observation records ({})'.format(
                    timestamp(), saved['formal_records'], saved['observed_records'], filename), flush=True)
                if saved['final_audit_saved'] or (destination / 'STOP').exists():
                    state['status'] = 'complete' if saved['final_audit_saved'] else 'stopped'
                    write_json(destination / 'service-state.json', state)
                    return
            except Exception as error:
                state['last_error'] = repr(error)
                print('{} snapshot error: {!r}'.format(timestamp(), error), flush=True)
            write_json(destination / 'service-state.json', state)
            # Short sleeps allow prompt final save once the independent audit arrives.
            next_save = start + interval
            while time.monotonic() < next_save:
                if ((run / 'outputs' / 'independent-audit.json').exists() or
                        (destination / 'STOP').exists()):
                    break
                time.sleep(min(5, max(0, next_save - time.monotonic())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        saved, name = snapshot(args.run, args.destination)
        print(json.dumps({'snapshot': name, 'formal_records': saved['formal_records']}))
    else:
        serve(args.run, args.destination, args.interval)
