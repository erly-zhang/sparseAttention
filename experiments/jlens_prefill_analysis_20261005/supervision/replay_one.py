"""Schedule one unchanged replay input; publish a shard only when all 20 exist."""
import argparse
import fcntl
import importlib.util
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))


def bind_local_aggregate():
    """Keep replay's finalizer bound to this run after legacy edits sys.path."""
    path = RUN / 'aggregate.py'
    existing = sys.modules.get('aggregate')
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location('aggregate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['aggregate'] = module
    return module


def publish_shard(directory, method, shard, items, read, save, sha):
    """The canonical completion marker always describes the full original shard."""
    expected = [item for item in items if item['analysis_shard'] == shard]
    with (directory / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = {}
        for item in expected:
            path = directory / (item['key'] + '.json')
            if not path.exists():
                return False
            row = read(path)
            if row['key'] != item['key'] or row['method'] != method:
                raise ValueError('Result belongs to a different input or method')
            if not (directory / row['artifact_file']).exists():
                raise ValueError('Missing saved activation artifact')
            files[path.name] = sha(path)
        save(directory / 'complete.json', dict(status='complete', method=method,
             shard=shard, records=len(expected), files=files, snapshot_error=None))
        return True


def run_one(method, index):
    # Only dispatch and completion bookkeeping change. The frozen replay.run
    # computes the identical forward pass, observations, metrics and artifacts.
    import replay
    from common import HERE, config, read, save, sha
    control = read(HERE / 'supervision/control.json')
    for path, expected in control['pinned_files'].items():
        if sha(path) != expected:
            raise ValueError('Pinned supervision source changed: ' + path)
    c = config()
    if method not in c['replay']['methods']:
        raise ValueError('Unknown method')
    original = read(HERE / 'input-manifest.json')
    items = original['evaluation']
    if not 0 <= index < len(items):
        raise ValueError('Invalid sample index')
    item = items[index]
    shard = item['analysis_shard']
    directory = HERE / 'outputs' / ('replay-' + method + '-' + str(shard))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (item['key'] + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = directory / (item['key'] + '.json')
        if result.exists():
            row = read(result)
            if sha(directory / row['artifact_file']) != row['artifact_sha256']:
                raise ValueError('Previous result artifact changed')
            publish_shard(directory, method, shard, items, read, save, sha)
            return
        original_read, original_save = replay.read, replay.save
        original_load = replay.load_model

        def guarded_load():
            import torch
            free, total = torch.cuda.mem_get_info()
            if free < 60 * 1024 ** 3:
                raise RuntimeError('Insufficient free allocated GPU memory: free={:.2f} GiB, total={:.2f} GiB'.format(
                    free / 1024 ** 3, total / 1024 ** 3))
            return original_load()

        def selected_read(path):
            data = original_read(path)
            if Path(path) == HERE / 'input-manifest.json':
                data = dict(data)
                data['evaluation'] = [item]
            return data

        def selected_save(path, value):
            path = Path(path)
            if path == directory / 'complete.json':
                publish_shard(directory, method, shard, items, read, save, sha)
            elif path == directory / 'failure.json':
                original_save(directory / (item['key'] + '.failure.json'), value)
            else:
                original_save(path, value)

        replay.read, replay.save, replay.load_model = selected_read, selected_save, guarded_load
        try:
            bind_local_aggregate()
            replay.run(method, shard)
        finally:
            replay.read, replay.save, replay.load_model = original_read, original_save, original_load


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--sample-index', type=int, required=True)
    args = p.parse_args()
    run_one(args.method, args.sample_index)
