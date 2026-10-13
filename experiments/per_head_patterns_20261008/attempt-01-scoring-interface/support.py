"""CPU-readable provenance and approval gates for a new isolated experiment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'representative_patterns_fast_retry03_20261001'
ORIGIN = HERE.parent / 'shared_topp_mask_bench_20260930'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def legacy():
    name = '_per_head_frozen_runner'
    if name not in sys.modules:
        sys.path.insert(0, str(OLD))
        spec = importlib.util.spec_from_file_location(name, OLD/'run.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def verify():
    p = read(HERE/'prepared.json')
    for path, expected in read(HERE/'lock.json')['files'].items():
        assert sha(path) == expected, path
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, path
    for path, expected in p['model_files'].items():
        s = Path(path).stat()
        assert s.st_size == expected['bytes'] and s.st_mtime_ns == expected['mtime_ns'], path
    return p


def gate(allocation=True):
    a = read(HERE/'approval.json')
    assert a.get('approved') is True and a['lock_sha256'] == sha(HERE/'lock.json')
    p = verify()
    if allocation:
        assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires Slurm'
    return p
