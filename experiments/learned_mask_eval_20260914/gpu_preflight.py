"""Selected-checkpoint and unchanged compact-kernel independent reference audit."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import unittest
import torch
from .checkpoints import audit_training, load
from .test_batched import Equivalence
from .selector import Selector
from experiments.learned_mask_nnonly_20260914.model import (
    VARIANTS, shared_masks, final_mask, exact_probability_mean)
from experiments.probmean_compact_20260906.selector import probability_mean
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
from experiments.learned_mask_nnonly_20260914.pipeline import ROOT, sha

OUT = ROOT/'experiments/outputs/learned_mask_eval_20260914'


def allocation():
    host = socket.gethostname().split('.')[0]
    print('HOST', host, flush=True)
    queue = subprocess.check_output(['squeue', '-u', 'jz648', '-h', '-o', '%i|%T|%N'], text=True)
    print(queue, flush=True)
    assert not host.startswith('login')
    job = os.environ.get('SLURM_JOB_ID')
    candidates = [line.split('|')[0] for line in queue.splitlines()
                  if line.split('|')[1] == 'RUNNING' and host in subprocess.check_output(
                      ['scontrol', 'show', 'hostnames', line.split('|')[2]], text=True).splitlines()]
    assert candidates and (job is None or job in candidates), 'No current compute allocation'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not active.strip(), 'GPU occupied: refusing test'
    return dict(host=host, allocation=job or candidates[0], queue=queue)


@torch.no_grad()
def gpu_checks(proof):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda'
    assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
    report = []
    for variant in VARIANTS:
        predictor = load(variant, proof, device)
        selector = Selector({'layers': predictor.groups}, predictor)
        selector.validate = True
        for n, layer in ((257, 0), (10497, 31)):
            q = torch.randn(1, 32, n, 128, dtype=torch.bfloat16, device=device).transpose(1, 2)
            k = torch.randn(1, 8, n, 128, dtype=torch.bfloat16, device=device).transpose(1, 2)
            v = torch.randn_like(k)
            token = selector.current_layer.set(layer)
            index = selector(q, k, v, 128, .95, 1, None)
            selector.current_layer.reset(token)
            output = triton_token_compacted_prefill_attention(q, k, v, index, token_chunk_size=32)
            assert torch.isfinite(output).all() and torch.stack(selector.validation).all()
            width, rows = min(n, 10240), (n+127)//128
            ids = index.token_indices.view(32, rows, width)
            error = 0.
            # Independent dense PyTorch QK over selected original keys, per query.
            for h in (0, 3, 4, 15, 31):
                for i in (0, 127, n-2, n-1):
                    keys = ids[h, i//128]
                    keys = keys[(keys <= i) & (keys < n)].long()
                    score = q[0, i, h].float() @ k[0, keys, h//4].float().T / math.sqrt(128)
                    reference = score.softmax(-1) @ v[0, keys, h//4].float()
                    torch.testing.assert_close(output[0, i, h].float(), reference, atol=.02, rtol=.02)
                    error = max(error, float((output[0, i, h].float()-reference).abs().max()))
            if n > 10240:
                a, b = n-1, n
                for group in predictor.groups[str(layer)]:
                    rep = group['representative']
                    p = probability_mean(q, k, rep, rep//4)[-1]
                    exact = exact_probability_mean(q[0, a:b, rep], k[0, :, rep//4], a)
                    torch.testing.assert_close(p, exact, atol=2e-6, rtol=.02)
                    protect, core, full = shared_masks(p, a, b, predictor.config)
                    for h in group['members']:
                        if h == rep:
                            continue
                        scores = predictor(q[0, a:b, h], k[0, :, h//4], p,
                            full if variant == 'full_mask' else core, layer, h)
                        expected = final_mask(scores, core, full, protect, b, predictor.config, variant)
                        actual = torch.zeros(n, device=device, dtype=torch.bool)
                        actual[ids[h, -1].long()] = True
                        # FP32 batching may perturb an almost-tied boundary. This
                        # gate intentionally fails rather than silently relaxing.
                        assert torch.equal(expected, actual), (variant, layer, h, int((expected ^ actual).sum()))
            stats = selector.stats.snapshot()
            report.append(dict(variant=variant, n=n, layer=layer, max_abs_error=error,
                               checkpoint_sha256=proof['selected'][variant]['checkpoint_sha256']))
            print('GPU_REFERENCE_PASS', report[-1], flush=True)
            del q, k, v, index, output
        del predictor, selector
        torch.cuda.empty_cache()
    return report


def main():
    access = allocation()
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = OUT/f'preflight_{time.time_ns()}'
    attempt.mkdir()
    code, error = 1, None
    try:
        proof = audit_training()
        (attempt/'training_audit.json').write_text(json.dumps(proof, indent=2))
        tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Equivalence))
        assert tests.wasSuccessful(), 'Batched scorer reference failed'
        report = gpu_checks(proof)
        sources = {str(p.resolve().relative_to(ROOT.resolve())): sha(p)
                   for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.sbatch')}
        dependencies = {name: sha(ROOT/name) for name in (
            'experiments/probmean_compact_20260906/selector.py',
            'experiments/probmean_compact_20260906/run.py',
            'experiments/residual6040_20260908/indices.py',
            'experiments/random_residual_20260908/random_policy.py',
            'experiments/token_compacted_sparse.py',
            'experiments/benchmark_shareprefill_ae3.py',
            'experiments/run_dense_multimodel_infinitebench.py',
            'experiments/shared_flexprefill_predictions_20260912/predictions.py')}
        (attempt/'correctness_passed.json').write_text(json.dumps(dict(status='passed', timestamp=time.time(),
            allocation=access, training_audit=proof, checks=report, source_sha256=sources,
            runtime_dependency_sha256=dependencies), indent=2))
        print('LEARNED_MASK_GPU_PREFLIGHT_PASSED', attempt, flush=True)
        code = 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        (attempt/'exit.json').write_text(json.dumps(dict(returncode=code, error=error, end_time=time.time())))


if __name__ == '__main__':
    main()
