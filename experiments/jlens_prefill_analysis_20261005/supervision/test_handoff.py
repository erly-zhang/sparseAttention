"""Verify the queued-job bridge yields without launching another cycle."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import controller_entry
import supervisor


class HandoffTests(unittest.TestCase):
    def test_bridge_yields_to_slurm_before_any_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor.save(root / 'state.json', dict(complete=False))
            supervisor.save(root / 'handoff-request.json', dict(job='123'))
            with patch.object(controller_entry, 'HERE', root), patch.object(supervisor, 'HERE', root), \
                    patch.object(controller_entry.subprocess, 'run') as run:
                controller_entry.bridge()
                run.assert_not_called()
            self.assertEqual(supervisor.read(root / 'bridge.json')['status'], 'handed_off')

    def test_finished_run_does_not_launch_another_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor.save(root / 'state.json', dict(complete=True))
            supervisor.save(root / 'control.json', {})
            with patch.object(controller_entry, 'HERE', root), patch.object(supervisor, 'HERE', root), \
                    patch.object(controller_entry, 'verify_control'), patch.object(controller_entry, 'command') as cmd:
                controller_entry.slurm()
                cmd.assert_not_called()


if __name__ == '__main__':
    unittest.main()
