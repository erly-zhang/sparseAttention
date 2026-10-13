"""CPU-only tests of task isolation, completion and retry ownership."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_one
import supervisor


class SupervisionTests(unittest.TestCase):
    def test_single_sample_adapter_preserves_results_and_full_shard_completion(self):
        import common
        import replay
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [dict(key='alpha', analysis_shard=0), dict(key='beta', analysis_shard=0)]
            common.save(root / 'input-manifest.json', dict(evaluation=items))
            common.save(root / 'config.json', dict(replay=dict(methods=['dense_full'])))
            common.save(root / 'supervision/control.json', dict(pinned_files={}))
            observed = []
            original_read, original_save, original_load = replay.read, replay.save, replay.load_model

            def fake_run(method, shard):
                selected = replay.read(root / 'input-manifest.json')['evaluation']
                self.assertEqual(len(selected), 1)
                item = selected[0]
                observed.append(item['key'])
                d = root / 'outputs/replay-dense_full-0'
                artifact = d / (item['key'] + '.pt')
                artifact.write_bytes(('unchanged-' + item['key']).encode())
                replay.save(d / (item['key'] + '.json'), dict(key=item['key'], method=method,
                    artifact_file=artifact.name, artifact_sha256=common.sha(artifact)))
                replay.save(d / 'complete.json', dict(records=1))

            with patch.object(common, 'HERE', root), patch.object(replay, 'run', fake_run):
                replay_one.run_one('dense_full', 0)
                complete = root / 'outputs/replay-dense_full-0/complete.json'
                self.assertFalse(complete.exists())
                replay_one.run_one('dense_full', 1)
                row = common.read(complete)
                self.assertEqual(row['records'], 2)
                self.assertEqual(set(row['files']), {'alpha.json', 'beta.json'})
                replay_one.run_one('dense_full', 0)  # Resume: no repeated model run.
                self.assertEqual(observed, ['alpha', 'beta'])
                artifact = root / 'outputs/replay-dense_full-0/alpha.pt'
                artifact.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    replay_one.run_one('dense_full', 0)
            self.assertIs(replay.read, original_read)
            self.assertIs(replay.save, original_save)
            self.assertIs(replay.load_model, original_load)

    def test_sparse_readiness_depends_on_its_own_dense_sample(self):
        tasks = {supervisor.identity(m, i): dict(done=False, attempts=[])
                 for m in ['dense_full', 'flexprefill'] for i in [0, 1]}
        tasks['dense_full:0']['done'] = True
        self.assertTrue(supervisor.ready('flexprefill', 0, tasks))
        self.assertFalse(supervisor.ready('flexprefill', 1, tasks))
        tasks['flexprefill:0']['blocked'] = {'error': 'scientific failure'}
        self.assertFalse(supervisor.ready('flexprefill', 0, tasks))

    def test_only_transient_failures_are_retryable(self):
        self.assertTrue(supervisor.retryable('FAILED', 'torch.OutOfMemoryError: CUDA out of memory'))
        self.assertTrue(supervisor.retryable('FAILED', 'Insufficient free allocated GPU memory'))
        self.assertTrue(supervisor.retryable('TIMEOUT', ''))
        self.assertFalse(supervisor.retryable('FAILED', 'Nonfinite readout'))
        self.assertFalse(supervisor.retryable('FAILED', 'Mixed configurations'))

    def test_submission_is_owned_before_qos_update_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = dict(roots=[], excluded_nodes=['w002'],
                         tasks={'dense_full:0': dict(done=False, attempts=[])})
            def fake_command(args):
                if args[0] == 'sbatch': return '12345\n'
                raise RuntimeError('temporary qos update failure')
            with patch.object(supervisor, 'HERE', root), patch.object(supervisor, 'command', fake_command):
                supervisor.submit('dense_full', [0], state)
            saved = supervisor.read(root / 'state.json')
            self.assertEqual(saved['roots'], ['12345'])
            self.assertEqual(saved['tasks']['dense_full:0']['attempts'][0]['job'], '12345_0')
            self.assertFalse(saved['tasks']['dense_full:0']['attempts'][0].get('handled', False))

    def test_scheduler_filters_unrelated_jobs_and_retains_accounting(self):
        def fake_command(args):
            if args[0] == 'sacct': return '123_0|COMPLETED|0:0|w005|\n'
            return '999|RUNNING|v006|None\n124_0|RUNNING|w010|None\n'
        with patch.object(supervisor, 'command', fake_command):
            states = supervisor.scheduler(['123', '124'])
        self.assertEqual(states['123_0']['state'], 'COMPLETED')
        self.assertEqual(states['124_0']['state'], 'RUNNING')
        self.assertNotIn('999', states)

    def test_source_change_stops_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f = root / 'source.py'; f.write_text('original')
            control = dict(pinned_files={str(f): supervisor.digest(f)})
            supervisor.save(root / 'approval.json', dict(approved=True))
            with patch.object(supervisor, 'RUN', root):
                supervisor.verify_control(control)
                f.write_text('modified')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    supervisor.verify_control(control)


if __name__ == '__main__':
    unittest.main()
