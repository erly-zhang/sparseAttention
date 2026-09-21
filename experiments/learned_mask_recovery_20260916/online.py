"""Unchanged inference with durable tokens after the existing timed generate."""
import json
import os
import types
from pathlib import Path
import torch
from experiments.learned_mask_eval_20260914 import online as old
from .journal import atomic_record, identity, read_record


class DurableRecorder(old.Recorder):
    def install(self):
        super().install()
        original, recorder = self.model.generate, self
        directory = Path(os.environ['LEARNED_RECOVERY_JOURNAL'])
        fingerprint = os.environ['LEARNED_RECOVERY_FINGERPRINT']
        cache = os.environ.get('LEARNED_RECOVERY_REPLAY')

        def generate(model, *args, **kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            assert ids is not None and ids.ndim == 2 and ids.shape[0] == 1
            key = old.input_ids_sha256(ids)
            cached = Path(cache) / (key + '.json') if cache else None
            if cached is not None:
                record = read_record(cached, fingerprint)
                assert identity(record['metric']) == (ids.shape[1], key)
                row = dict(record['metric'], call_index=len(recorder.rows))
                recorder.rows.append(row)
                recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
                suffix = torch.tensor(record['generated_ids'], device=ids.device, dtype=ids.dtype)[None]
                result = torch.cat((ids, suffix), dim=1)
            else:
                result = original(*args, **kwargs)
                assert torch.equal(result[:, :ids.shape[1]], ids)
                row = dict(recorder.rows[-1])
                assert identity(row) == (ids.shape[1], key)
                record = dict(fingerprint=fingerprint, metric=row,
                    generated_ids=result[0, ids.shape[1]:].detach().cpu().tolist(),
                    origin_job_id=os.environ.get('SLURM_JOB_ID'),
                    origin_host=os.uname().nodename,
                    format='original generated token IDs before HFLM stop/postprocessing')
            atomic_record(directory, record)
            return result
        self.model.generate = types.MethodType(generate, self.model)


old.base.runner.GenerationMetricsRecorder = DurableRecorder

if __name__ == '__main__':
    old.base.runner.main()
