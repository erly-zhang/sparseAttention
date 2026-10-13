import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from snapshot import snapshot


class SnapshotTests(unittest.TestCase):
    def test_diagnostic_observations_save_all_verified_repeat_logits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            worker = root / 'run' / 'outputs' / 'worker-0'
            worker.mkdir(parents=True)
            passes = []
            for repeat in range(2):
                data = ('repeat-' + str(repeat)).encode()
                name = str(repeat) + '.npy'
                (worker / name).write_bytes(data)
                passes.append({'logits_file': name, 'logits_sha256': hashlib.sha256(data).hexdigest()})
            (worker / 'sample.json').write_text(json.dumps({'key': 'sample', 'task': 'qa', 'layers': {}, 'passes': passes}))
            saved, _ = snapshot(root / 'run', root / 'backup')
            self.assertEqual(saved['observed_records'], 1)
            self.assertEqual(saved['formal_records'], 0)
            self.assertEqual(saved['observations_by_task'], {'qa': 1})
            self.assertEqual(len(saved['files']), 3)

    def test_service_saves_on_timer_and_flushes_before_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, backup = root / 'run', root / 'backup'
            (run / 'outputs').mkdir(parents=True)
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('snapshot.py')),
                '--run', str(run), '--destination', str(backup), '--interval', '.05'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while len(list((backup / 'snapshots').glob('*.json'))) < 2:
                    if time.monotonic() > deadline or process.poll() is not None:
                        self.fail('Snapshot timer did not produce two snapshots')
                    time.sleep(.02)
                (backup / 'STOP').touch()
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr.decode())
                state = json.loads((backup / 'service-state.json').read_text())
                self.assertEqual(state['status'], 'stopped')
                self.assertGreaterEqual(len(list((backup / 'snapshots').glob('*.json'))), 3)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_verified_results_are_recoverable_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, backup = root / 'run', root / 'backup'
            worker = run / 'outputs' / 'formal-0'
            worker.mkdir(parents=True)
            logits = b'complete logits'
            (worker / 'sample.npy').write_bytes(logits)
            record = {'prediction': 'A', 'method': 'dense_full', 'task': 'qa',
                      'score': 1., 'logits_file': 'sample.npy',
                      'logits_sha256': hashlib.sha256(logits).hexdigest()}
            original = json.dumps(record).encode()
            (worker / 'sample.json').write_bytes(original)
            first, first_name = snapshot(run, backup)
            second, second_name = snapshot(run, backup)
            self.assertEqual(first['formal_records'], 1)
            self.assertEqual(first['by_method'], {'dense_full': 1})
            self.assertEqual(first['files'], second['files'])
            self.assertNotEqual(first_name, second_name)
            self.assertEqual(len([p for p in (backup / 'objects').rglob('*') if p.is_file()]), 2)
            for name, data in [('sample.json', original), ('sample.npy', logits)]:
                info = first['files']['outputs/formal-0/' + name]
                self.assertEqual((backup / info['object']).read_bytes(), data)
            self.assertEqual((worker / 'sample.json').read_bytes(), original)

    def test_partial_or_mismatched_logits_do_not_count_as_saved_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, backup = root / 'run', root / 'backup'
            worker = run / 'outputs' / 'formal-0'
            worker.mkdir(parents=True)
            (worker / 'sample.npy').write_bytes(b'incomplete')
            (worker / 'sample.json').write_text(json.dumps({
                'prediction': 'A', 'method': 'dense', 'task': 'qa', 'score': 1,
                'logits_file': 'sample.npy', 'logits_sha256': 'incorrect'}))
            (worker / 'partial.json').write_text('{')
            (worker / 'failure.json').write_text('{"message":"failed"}')
            saved, _ = snapshot(run, backup)
            self.assertEqual(saved['formal_records'], 0)
            self.assertEqual(len(saved['skipped_files']), 2)
            self.assertEqual(saved['failure_files'], ['outputs/formal-0/failure.json'])
            self.assertNotIn('outputs/formal-0/sample.npy', saved['files'])

    def test_final_audit_is_saved_and_marks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run' / 'outputs').mkdir(parents=True)
            (root / 'run' / 'outputs' / 'independent-audit.json').write_text('{"status":"PASS"}')
            saved, _ = snapshot(root / 'run', root / 'backup')
            self.assertTrue(saved['final_audit_saved'])
            with self.assertRaises(ValueError):
                snapshot(root / 'run', root / 'run' / 'outputs' / 'backup')


if __name__ == '__main__':
    unittest.main()
