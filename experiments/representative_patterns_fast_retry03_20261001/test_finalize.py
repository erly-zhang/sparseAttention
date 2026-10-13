"""Test last-worker election without GPU, scheduler, or checkpoint loading."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from finalize import finalize_if_ready


class FinalizeTests(unittest.TestCase):
    def test_incomplete_peers_exit_and_last_worker_aggregates_once(self):
        config={'calibration_job':{'gpus':3},'formal':{'gpus':70}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs').mkdir()
            (root/'source-lock.json').write_text('{}')
            with patch('finalize.RUN',root),patch('aggregate.aggregate') as aggregate:
                for index in range(3):
                    folder=root/'outputs'/f'calibration-{index}';folder.mkdir()
                    (folder/'complete.json').write_text(json.dumps({'status':'complete'}))
                    result=finalize_if_ready('calibration',config)
                    self.assertEqual(result,index==2)
                aggregate.assert_called_once_with(require_hidden=False)
                self.assertFalse(finalize_if_ready('calibration',config))
                aggregate.assert_called_once()

    def test_failed_peer_prevents_postprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs').mkdir()
            folder=root/'outputs/calibration-0';folder.mkdir()
            (folder/'complete.json').write_text(json.dumps({'status':'complete'}))
            (folder/'failure.json').write_text('{}')
            with patch('finalize.RUN',root),patch('aggregate.aggregate') as aggregate:
                self.assertFalse(finalize_if_ready('calibration',{'calibration_job':{'gpus':1},'formal':{'gpus':1}}))
                aggregate.assert_not_called()


if __name__=='__main__':
    unittest.main()
