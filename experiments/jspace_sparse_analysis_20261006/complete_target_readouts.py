"""Complete target-piece ranks from saved residuals, without a model forward.

Uses the existing J-lens, BF16 RMSNorm / unembedding, full-vocabulary argsort,
and original 16-position readout chunks. Execution stays inside a Slurm GPU
allocation. Writes separate, resumable post-processing outputs.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import torch
from safetensors import safe_open
from jlens.vis import _ranks_of

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'jlens_prefill_analysis_20261005'
OUT = HERE / 'review/target_value_readouts'
METHODS = ['dense_full', 'coverage_token_vertical_slash', 'flexprefill']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, allow_nan=False))
    temp.replace(path)


def rank_bounds(logits, ids):
    selected = logits.gather(1, ids)
    best = torch.ones_like(ids)
    worst = torch.zeros_like(ids)
    for start in range(0, logits.shape[1], 16384):
        piece = logits[:, None, start:start+16384]
        best += (piece > selected[:, :, None]).sum(-1)
        worst += (piece >= selected[:, :, None]).sum(-1)
    return best, worst


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU post-processing requires Slurm'
    assert torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info()[0] > 6 * 1024**3
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(HERE / 'manifest.json')
    payload = read(HERE / 'review/figure5_prefill_data.json')
    target_ids = sorted(set(payload['target_ids']))
    assert len(target_ids) == 18
    key = payload['key']
    rows = {r['method']: r for r in manifest['records'] if r['key'] == key and r['method'] in METHODS}
    assert set(rows) == set(METHODS)
    inputs = {}
    artifacts = {}
    for method, row in rows.items():
        path = Path(row['artifact'])
        assert sha(path) == row['artifact_sha256']
        inputs[str(path)] = row['artifact_sha256']
        artifacts[method] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert artifacts[method]['positions'] == payload['positions']
    lens_path = Path(manifest['lens'])
    assert sha(lens_path) == manifest['lens_sha256']
    inputs[str(lens_path)] = manifest['lens_sha256']
    lens = torch.load(lens_path, map_location='cpu', weights_only=True, mmap=True)
    model = Path(manifest['model'])
    config = read(model / 'config.json')
    weight_map = read(model / 'model.safetensors.index.json')['weight_map']
    for name in ['lm_head.weight', 'model.norm.weight']:
        path = model / weight_map[name]
        expected = manifest['metadata'].get(str(path))
        if expected:
            assert sha(path) == expected['sha256']
            inputs[str(path)] = expected['sha256']
    def weight(name):
        with safe_open(str(model / weight_map[name]), framework='pt', device='cpu') as f:
            return f.get_tensor(name).to('cuda')
    w = weight('lm_head.weight')
    gamma = weight('model.norm.weight')
    assert w.dtype == gamma.dtype == torch.bfloat16
    eps = config['rms_norm_eps']
    targets = torch.tensor(target_ids, device='cuda', dtype=torch.long)
    identity = dict(key=key, methods=METHODS, target_ids=target_ids, positions=payload['positions'],
                    source_sha256=inputs, definition='one-based full-vocabulary argsort ranks; tie bounds also recorded',
                    arithmetic='FP32 J-lens transport; BF16 RMSNorm input/output and unembedding; TF32 disabled',
                    readout_chunk_positions=16, model_forward=False, generation=False,
                    j_lens_refit=False, gpu=torch.cuda.get_device_name(0), job=os.environ['SLURM_JOB_ID'])
    save(OUT / 'provenance.json', identity)
    if (OUT / 'complete.json').exists():
        print('TARGET RANKS ALREADY COMPLETE', flush=True)
        return
    started = time.monotonic()
    n = len(payload['positions'])
    final_index = payload['positions'].index(payload['input_tokens'] - 1)
    reference_cache_checks = {}
    for layer in range(32):
        path = OUT / f'layer-{layer}.json'
        if path.exists():
            continue
        matrix = lens['J'][layer].to(device='cuda', dtype=torch.float32) if layer != 31 else None
        results = {}
        for method in METHODS:
            h = artifacts[method]['residuals'][layer].float().cuda()
            if matrix is not None:
                h = h @ matrix.T
            fields = {name: [] for name in ['ranks', 'rank_best', 'rank_worst', 'log_probabilities']}
            compatible, count = 0, 0
            for start in range(0, n, 16):
                hidden = h[start:start+16].to(torch.bfloat16)
                normalized = hidden.float()
                normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps)
                normalized = gamma * normalized.to(torch.bfloat16)
                logits = torch.nn.functional.linear(normalized, w).float()
                assert torch.isfinite(logits).all()
                rank = _ranks_of(logits, targets) + 1
                selected_ids = targets[None, :].expand(logits.shape[0], -1)
                best, worst = rank_bounds(logits, selected_ids)
                assert ((rank >= best) & (rank <= worst)).all()
                logp = logits.log_softmax(-1).index_select(1, targets)
                for name, tensor in [('ranks', rank), ('rank_best', best), ('rank_worst', worst), ('log_probabilities', logp)]:
                    fields[name].extend(tensor.cpu().tolist())
                cached_ids = torch.tensor(payload['dense_ids'][layer][start:start+16], device='cuda')
                cached_ranks = torch.tensor(payload['method_data'][method]['reference_ranks'][layer][start:start+16], device='cuda')
                cbest, cworst = rank_bounds(logits, cached_ids)
                compatible += int(((cached_ranks >= cbest) & (cached_ranks <= cworst)).sum())
                count += cached_ranks.numel()
            reference_cache_checks[method] = dict(compatible=compatible, total=count)
            fields['legacy_rank_tie_compatibility'] = dict(compatible=compatible, total=count, fraction=compatible/count)
            fields['complete'] = True
            results[method] = fields
            # The pinned ca trace that motivated this page must remain compatible
            # with the independently recorded original ranks where available.
            legacy = payload['first_piece_final_position_full_rank_trace'][method][layer]
            if legacy is not None:
                j = target_ids.index(payload['target_ids'][0])
                assert fields['rank_best'][final_index][j] <= legacy <= fields['rank_worst'][final_index][j], (method, layer, legacy)
        save(path, dict(layer=layer, target_ids=target_ids, methods=results))
        save(OUT / 'progress.json', dict(completed_layers=len(list(OUT.glob('layer-*.json'))), total_layers=32,
                                        last_layer=layer, seconds=time.monotonic()-started))
        print('COMPLETE TARGET READOUT LAYER', layer, round(time.monotonic()-started, 1), flush=True)
        del matrix
    assert len(list(OUT.glob('layer-*.json'))) == 32
    save(OUT / 'complete.json', dict(status='PASS', layers=32, methods=3, positions=n,
                                    unique_target_tokens=18, token_occurrences_in_value=25,
                                    model_forward=False, generation=False, j_lens_refit=False,
                                    files={p.name: sha(p) for p in sorted(OUT.glob('layer-*.json'))}))
    print('FULL TARGET READOUTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
