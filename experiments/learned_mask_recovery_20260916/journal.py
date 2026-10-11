"""Atomic per-input generation records; no CUDA or model dependencies."""
import hashlib
import json
import os
from pathlib import Path


def identity(row):
    return row['input_tokens'], row['input_ids_sha256']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_record(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(record, record_sha256=digest(record))
    final = directory / (record['metric']['input_ids_sha256'] + '.json')
    temporary = directory / (final.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    # link is exclusive and atomic: never replace a previous completed input.
    os.link(temporary, final)
    temporary.unlink()
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_record(path, fingerprint):
    value = json.loads(Path(path).read_text())
    checksum = value.pop('record_sha256')
    assert digest(value) == checksum, 'Generation journal checksum mismatch'
    assert value['fingerprint'] == fingerprint, 'Generation provenance mismatch'
    assert 0 < len(value['generated_ids']) <= 128
    assert all(isinstance(x, int) and x >= 0 for x in value['generated_ids'])
    return value


def shard_indices(shard, count=497, shards=3):
    assert 0 <= shard < shards
    return list(range(shard, count, shards))
