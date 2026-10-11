import json
from pathlib import Path
import tempfile
import unittest
from .journal import atomic_record, read_record, shard_indices


class JournalTests(unittest.TestCase):
    def test_partition(self):
        shards = [shard_indices(i) for i in range(3)]
        self.assertEqual(list(map(len, shards)), [166, 166, 165])
        self.assertEqual(sorted(sum(shards, [])), list(range(497)))
        self.assertEqual(len(set(sum(shards, []))), 497)

    def test_atomic_roundtrip_no_overwrite_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            row = dict(fingerprint='proof', metric=dict(input_tokens=2, input_ids_sha256='abc'),
                       generated_ids=[2, 3])
            atomic_record(folder, row)
            path = Path(folder)/'abc.json'
            self.assertEqual(read_record(path, 'proof'), row)
            with self.assertRaises(FileExistsError):
                atomic_record(folder, row)
            with self.assertRaises(AssertionError):
                read_record(path, 'changed-proof')
            value = json.loads(path.read_text())
            value['generated_ids'] = [9]
            path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):
                read_record(path, 'proof')


if __name__ == '__main__':
    unittest.main()
