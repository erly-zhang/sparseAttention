"""Independent CPU Philox and FP32 attention references; GPU gates, no LLM data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

from experiments.random_score_residual_20260910.sampler import masked_scores, seed_values, random_score_residual
from experiments.random_score_residual_20260910.selector import Selector, LegacySelector
from experiments.residual6040_20260908.indices import shared_indices, assemble, actual_counts
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention

HERE = Path(__file__).resolve().parent
RELATIVE_LIMIT, ABSOLUTE_LIMIT = .005, .02


def cpu_philox(seed, offsets):
    """Philox4x32-10, written independently with NumPy uint64 intermediates."""
    mask = np.uint64(0xffffffff)
    c0 = np.asarray(offsets, dtype=np.uint64)
    c1, c2, c3 = (np.zeros_like(c0) for _ in range(3))
    unsigned = int(seed) & ((1 << 64) - 1)
    k0, k1 = unsigned & 0xffffffff, unsigned >> 32
    for _ in range(10):
        a, b = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((b >> np.uint64(32)) ^ c1 ^ np.uint64(k0), b & mask,
                          (a >> np.uint64(32)) ^ c3 ^ np.uint64(k1), a & mask)
        k0 = (k0 + 0x9E3779B9) & 0xffffffff
        k1 = (k1 + 0xBB67AE85) & 0xffffffff
    return c0


def score_reference(core, end, seed):
    n = len(core)
    j = np.arange(n, dtype=np.uint64)
    primary = cpu_philox(seed, j) >> np.uint64(8)
    secondary = cpu_philox(seed, j + np.uint64(1048576)) >> np.uint64(10)
    # Independent lexicographic tuple ordering, not torch.topk.
    candidates = [i for i in range(end) if not core[i]]
    order = sorted(candidates, key=lambda i: (-int(primary[i]), -int(secondary[i]), i))
    code = ((((primary << np.uint64(22)) | secondary) << np.uint64(17)) |
            (np.uint64(131071) - j)).astype(np.int64)
    code[np.asarray(core) | (np.arange(n) >= end)] = -1
    return code, order


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def fingerprint(device):
    # Fixed core avoids hardware-dependent floating point in random-stream audit.
    n, rows = 10497, 83
    p = torch.zeros((rows, n), device=device)
    b = shared_indices(p)
    fixed = torch.arange(6144, device=device).int()[None, :].expand(rows, -1).contiguous()
    b['core'] = fixed
    b['mask'].zero_()
    b['mask'][:, :6144] = True
    values = []
    for head in (1, 7, 31):
        # Core construction uses historical tie handling; compare sampling only
        # on a deterministic explicitly specified core for cross-device checks.
        r = random_score_residual(b, 'a' * 64, 42, 17, head)
        final = assemble(b, r)
        values.append(final.cpu().numpy().tobytes())
    return hashlib.sha256(b''.join(values)).hexdigest()


def run(device):
    checks = []
    # Seeded primitives against CPU Philox, global TopK and block-size invariance.
    n = 641
    rng = np.random.default_rng(31)
    core_cpu = rng.random((6, n)) < .3
    ends_cpu = np.array([17, 127, 128, 257, 512, 641])
    seeds_cpu = seed_values('1' * 64, 42, 5, 13, range(6))
    core = torch.tensor(core_cpu, device=device)
    ends = torch.tensor(ends_cpu, device=device)
    seeds = torch.tensor(seeds_cpu, dtype=torch.int64, device=device)
    scores = masked_scores(core, ends, seeds)
    assert torch.equal(scores, masked_scores(core, ends, seeds, block=128))
    for i in range(6):
        expected, order = score_reference(core_cpu[i], int(ends_cpu[i]), seeds_cpu[i])
        np.testing.assert_array_equal(scores[i].cpu().numpy(), expected)
        count = min(7, len(order))
        assert scores[i].topk(count).indices.tolist() == order[:count]
    checks.append('GPU Philox matches independent CPU; pre-TopK exclusions; global lexicographic TopK; block invariance')
    # Deliberate primary/secondary ties exercise documented hierarchy.
    primary = torch.tensor([7, 7, 7, 6], device=device, dtype=torch.int64)
    secondary = torch.tensor([1, 9, 9, 99], device=device, dtype=torch.int64)
    code = ((primary << 22 | secondary) << 17) | (131071 - torch.arange(4, device=device))
    assert code.topk(4).indices.tolist() == [1, 2, 0, 3]
    checks.append('finite precision ties: independent secondary then smaller original position')

    # 4096 local draws are a sampler sanity check, not full model seed experiments.
    trials, candidates, keep = 4096, 23, 7
    cm = torch.zeros((trials, candidates + 5), device=device, dtype=torch.bool)
    cm[:, :5] = True
    se = torch.tensor(seed_values('frequency-fixture', 42, 0, 1, range(trials)), device=device)
    en = torch.full((trials,), candidates + 5, device=device, dtype=torch.int64)
    top = masked_scores(cm, en, se).topk(keep, dim=-1).indices.cpu().numpy() - 5
    hist = np.bincount(top.ravel(), minlength=candidates)
    expected = trials * keep / candidates
    sd = np.sqrt(trials * (keep / candidates) * (1 - keep / candidates))
    assert np.max(np.abs(hist - expected)) < 6 * sd
    checks.append('marginal frequency within predefined 6-sigma bound')

    config = {'layers': {'0': [{'representative': 0, 'members': [0, 1]},
                                {'representative': 2, 'members': [2, 3]}]}}
    maximum_rel, maximum_abs = 0., 0.
    cpu_rng = torch.Generator(device='cpu').manual_seed(73)
    for n in (1, 127, 128, 129, 255, 257, 511, 512, 513, 641, 1025):
        q = torch.randn((1, 4, n + 9, 128), generator=cpu_rng).to(device, torch.bfloat16).transpose(1, 2)[:, :n]
        k = torch.randn((1, 2, n + 9, 128), generator=cpu_rng).to(device, torch.bfloat16).transpose(1, 2)[:, :n]
        v = torch.randn((1, 2, n + 9, 128), generator=cpu_rng).to(device, torch.bfloat16).transpose(1, 2)[:, :n]
        for interleave in (False, True):
            new, old = Selector(config, budget=512, core=256), LegacySelector(config, 'random', budget=512, core=256)
            for sel in (new, old):
                sel.current_layer.set(0)
                sel.capture = True
                sel.input_hash = 'test-input'
            before_cpu, before_gpu = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            before_python, before_numpy = random.getstate(), np.random.get_state()
            idx = new(q, k, v, 128, .95, 1, None, gqa_interleave=interleave)
            assert torch.equal(before_cpu, torch.get_rng_state())
            assert torch.equal(before_gpu, torch.cuda.get_rng_state(device))
            assert random.getstate() == before_python
            after_numpy = np.random.get_state()
            assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
            np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
            out = triton_token_compacted_prefill_attention(q, k, v, idx, token_chunk_size=32, gqa_interleave=interleave)
            old(q, k, v, 128, .95, 1, None, gqa_interleave=interleave)
            for row, previous in zip(new.captured, old.captured):
                _, h, rep, core_idx, final, shared, _ = row
                assert torch.equal(core_idx, previous[3]) and torch.equal(shared, previous[5])
                if h == rep: assert torch.equal(final, previous[4])
                assert torch.equal(actual_counts(final, n), actual_counts(previous[4], n))
                mask = torch.zeros((len(final), n), dtype=torch.bool, device=device)
                for tile, f in enumerate(final):
                    end = min((tile + 1) * 128, n)
                    ids = f[f < end]
                    c = core_idx[tile][core_idx[tile] < end]
                    assert len(ids) == min(512, end) and len(ids.unique()) == len(ids)
                    assert torch.all(ids[1:] > ids[:-1])
                    assert set(c.tolist()) <= set(ids.tolist())
                    protected = set(range(min(128, end))) | set(range(tile * 128, end))
                    assert protected <= set(ids.tolist())
                    residual = set(ids.tolist()) - set(c.tolist())
                    assert not residual & set(c.tolist())
                    if h != rep: assert len(residual) == min(512, end) - min(256, end)
                    if end <= 512: assert ids.tolist() == list(range(end))
                    mask[tile, ids.long()] = True
                pos = torch.arange(n, device=device)
                causal = mask[pos // 128] & (pos[:, None] >= pos[None, :])
                kh = h % 2 if interleave else h // 2
                logits = q[0, :, h].float() @ k[0, :, kh].float().T / np.sqrt(128.)
                ref = logits.masked_fill(~causal, -torch.inf).softmax(-1) @ v[0, :, kh].float()
                delta = out[0, :, h].float() - ref
                rel, absolute = float(delta.norm() / ref.norm()), float(delta.abs().max())
                maximum_rel, maximum_abs = max(maximum_rel, rel), max(maximum_abs, absolute)
                assert rel <= RELATIVE_LIMIT and absolute <= ABSOLUTE_LIMIT, (n, rel, absolute)
        print('PASS tensor length', n, flush=True)
    checks.append('old core bitwise equality, both GQA mappings, protection, budgets, partial/padded tiles, pair counts, independent FP32 masked attention')
    checks.append('global Python/NumPy/CPU/CUDA RNG unchanged')
    # Exact production head/group/GQA scope on small tensors, no language model.
    config32 = {'layers': {'0': [{'representative': r, 'members': list(range(r, 32, 3))} for r in range(3)]}}
    q = torch.randn((1, 641, 32, 128), generator=cpu_rng).to(device, torch.bfloat16)
    k = torch.randn((1, 641, 8, 128), generator=cpu_rng).to(device, torch.bfloat16)
    v = torch.randn((1, 641, 8, 128), generator=cpu_rng).to(device, torch.bfloat16)
    sel = Selector(config32, budget=512, core=256)
    sel.current_layer.set(0)
    sel.capture, sel.validate = True, True
    sel(q,k,v,128,.95,1,None)
    assert len(sel.captured) == len(sel.validation) == 32 and torch.stack(sel.validation).all()
    assert sum(h != rep for _,h,rep,*_ in sel.captured) == 29
    for _,h,rep,_,final,old,_ in sel.captured:
        if h == rep: assert torch.equal(final,old)
    checks.append('32 Q heads / 8 KV heads: all 29 nonrepresentatives active, three representatives unchanged')
    # Production budgets and scores tied everywhere: exact old core tie policy.
    p = torch.zeros((83, 10497), device=device)
    b = shared_indices(p)
    r = random_score_residual(b, 'production-budget', 42, 3, 5)
    assert torch.equal(r, random_score_residual(b, 'production-budget', 42, 3, 5, tile_batch=1, block=128))
    final = assemble(b, r)
    assert torch.equal(actual_counts(final, 10497), actual_counts(b['old'], 10497))
    for tile in (80, 81, 82):
        core_set, res_set = set(b['core'][tile].tolist()), set(r[tile].tolist())
        assert len(core_set) == 6144 and len(res_set) == 4096 and not core_set & res_set
        assert max(res_set) < int(b['ends'][tile])
    checks.append('production 6144+4096 budgets, deterministic ties, tile batching/relaunch invariance')
    return {'passed': True, 'checks': checks, 'max_relative_l2': maximum_rel,
            'max_absolute_error': maximum_abs, 'relative_limit': RELATIVE_LIMIT,
            'absolute_limit': ABSOLUTE_LIMIT, 'frequency_trials': trials,
            'frequency_candidates': candidates, 'frequency_kept': keep,
            'frequency_histogram': hist.tolist(), 'frequency_expected': expected,
            'frequency_max_z': float(np.max(np.abs(hist - expected)) / sd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fingerprint-only', action='store_true')
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no fake CPU pass')
    result = {} if args.fingerprint_only else run(torch.device('cuda:0'))
    result.update({'fingerprint': fingerprint(torch.device('cuda:0')), 'hostname': platform.node(),
        'hardware': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.version,
        'cross_gpu_passed': False,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}})
    result['legacy_source_sha256'] = {
        str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE.parent/'residual6040_20260908/indices.py',
                  HERE.parent/'residual6040_20260908/selector6040.py',
                  HERE.parent/'probmean_compact_20260906/selector.py',
                  HERE.parent/'token_compacted_sparse.py']}
    write_new(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
