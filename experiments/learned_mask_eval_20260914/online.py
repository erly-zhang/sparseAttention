"""Canonical complete generation with frozen learned masks and wall-clock timing."""
import os
import json
import sys
import time
import types
from pathlib import Path
import torch
from .checkpoints import load, read
from .selector import Selector
from experiments.learned_mask_nnonly_20260914.model import VARIANTS
from experiments.benchmark_shareprefill_ae3 import input_ids_sha256

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'probmean_compact_20260906'))
import run as base

METHODS = {f'nnonly_{v}_compact10240': v for v in VARIANTS}
base.runner.METHODS = tuple(base.runner.METHODS) + tuple(METHODS)
base.runner.SHAREPREFILL_METHODS = tuple(base.runner.SHAREPREFILL_METHODS) + tuple(METHODS)
VALIDATE = os.environ.get('LEARNED_MASK_SMOKE') == '1'
PROOF = read(os.environ['LEARNED_MASK_TRAINING_AUDIT'])


def install(model, method, **kwargs):
    variant = METHODS[method]
    device = next(model.parameters()).device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.random.fork_rng(devices=[device]):
        predictor = load(variant, PROOF, device)
    original = base.sparse.RepresentativeTokenFirstBlockSelector
    class Learned(Selector):
        def __init__(self, config, **unused):
            super().__init__(config, predictor)
    base.sparse.RepresentativeTokenFirstBlockSelector = Learned
    try:
        result, patch, _ = base.old_install(model, 'shareprefill_ae3_token_compact', **kwargs)
    finally:
        base.sparse.RepresentativeTokenFirstBlockSelector = original
    sel = patch.selector
    # Exercise learned paths and partial-tile kernel compilation before timing.
    with torch.random.fork_rng(devices=[device]), torch.no_grad():
        for n in (10496, 10497):
            q = torch.randn(1, 32, n, 128, device=device, dtype=torch.bfloat16).transpose(1, 2)
            k = torch.randn(1, 8, n, 128, device=device, dtype=torch.bfloat16).transpose(1, 2)
            v = torch.randn_like(k)
            token = sel.current_layer.set(0)
            index = sel(q, k, v, 128, .95, 1, None)
            output = base.sparse.triton_token_compacted_prefill_attention(q, k, v, index, token_chunk_size=32)
            assert torch.isfinite(output).all()
            sel.current_layer.reset(token)
        del q, k, v, index, output
    torch.cuda.synchronize()
    sel.stats = type(sel.stats)()
    sel.validate = VALIDATE
    return result, patch, dict(implementation=method, num_groups_per_layer=3,
        selected_checkpoint=PROOF['selected'][variant], grouping_independent_of_validation=False,
        online_config=dict(query_tile=128, score='mean_of_per_real_query_causal_softmax',
            final_budget=10240, core_budget=None if variant == 'full_mask' else 6144,
            sink=128, local='current real query tile', global_tail=False, block_projection=False,
            representative_selection='unchanged historical Shared10240 including topk ties',
            member_condition_selection='stable position ties matching NN training',
            nn_dtype='float32', model_dtype='bfloat16', representatives=3, members=29),
        timing_scope='GPU synchronized whole prefill, including NN, selection, indexing, counts and attention')


class Recorder(base.Recorder.__bases__[0]):
    def install(self):
        super().install()
        inner, recorder = self.model.generate, self
        def generate(model, *args, **kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            identity = input_ids_sha256(ids)
            sel = recorder.patch.selector
            sel.validation.clear()
            stamp, finite = {}, []
            def before(_m, _a, named):
                x = named.get('input_ids')
                if 'start' not in stamp and x is not None and x.shape[1] > 1:
                    torch.cuda.synchronize()
                    stamp['start'] = time.perf_counter()
            def after(_m, _a, named, out):
                if 'start' in stamp and 'end' not in stamp:
                    torch.cuda.synchronize()
                    stamp['end'] = time.perf_counter()
                if VALIDATE and hasattr(out, 'logits'):
                    finite.append(torch.isfinite(out.logits).all())
            hooks = [model.register_forward_pre_hook(before, with_kwargs=True),
                     model.register_forward_hook(after, with_kwargs=True)]
            try:
                result = inner(*args, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            row = recorder.rows[-1]
            assert row['input_ids_sha256'] == identity and result.shape[1] > ids.shape[1]
            if VALIDATE:
                assert sel.validation and torch.stack(sel.validation).all()
                assert finite and torch.stack(finite).all()
                row['smoke_indices_and_finite_logits_passed'] = True
            row['prefill_cuda_event_sec'] = row['prefill_latency_sec']
            row['prefill_latency_sec'] = stamp['end'] - stamp['start']
            row['decode_latency_sec'] = row['total_latency_sec'] - row['prefill_latency_sec']
            row['timing_scope'] = 'synchronized whole prefill, including frozen NN and indexing'
            recorder.output_path.write_text(''.join(json.dumps(r)+'\n' for r in recorder.rows))
            return result
        self.model.generate = types.MethodType(generate, self.model)


base.runner.install_benchmark_method = install
base.runner.GenerationMetricsRecorder = Recorder
def evaluate(*args, **kwargs):
    kwargs['log_samples'] = True
    return base.original_evaluate(*args, **kwargs)
base.lm_eval.simple_evaluate = evaluate

if __name__ == '__main__':
    base.runner.main()
