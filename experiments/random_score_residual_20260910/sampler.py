"""Stateless per-key Uniform priorities, followed by global TopK per tile."""
from contextlib import nullcontext
import hashlib
import struct

import torch
import triton
import triton.language as tl

STREAM = 'random-score-residual-v1'
POSITION_BITS = 17
MAX_CONTEXT = 1 << POSITION_BITS


def seed_values(input_hash, seed, layer, head, tiles):
    prefix = f'{STREAM}|{seed}|{input_hash}|{layer}|{head}|'.encode()
    return [struct.unpack('<q', hashlib.sha256(prefix + str(int(r)).encode()).digest()[:8])[0]
            for r in tiles]


@triton.jit
def _scores(Seeds, Core, Ends, Out, N:tl.constexpr, BLOCK:tl.constexpr):
    r, c = tl.program_id(0), tl.program_id(1)
    j = c * BLOCK + tl.arange(0, BLOCK)
    seed = tl.load(Seeds + r).to(tl.uint64)
    # Separate Philox counters: primary uniform and independent tie uniform.
    primary = tl.randint(seed, j.to(tl.uint32)).to(tl.uint32) >> 8
    secondary = tl.randint(seed, (j + 1048576).to(tl.uint32)).to(tl.uint32) >> 10
    # Conceptual u=(primary+0.5)/2**24; retain its exact integer ordering.
    # 24 primary + 22 independent tie + 17 deterministic position bits = 63.
    code = ((primary.to(tl.int64) << 22) | secondary.to(tl.int64)) << 17
    code = code | (131071 - j.to(tl.int64))
    end = tl.load(Ends + r)
    in_core = tl.load(Core + r * N + j, j < N, 0)
    legal = (j < N) & (j < end) & ~in_core
    tl.store(Out + r * N + j, tl.where(legal, code, -1), j < N)


def masked_scores(core_mask, ends, seeds, block=256):
    """Materialize all original key priorities; exclusions happen before TopK."""
    rows, n = core_mask.shape
    if not 0 < n <= MAX_CONTEXT:
        raise ValueError('Priority encoding supports context lengths 1..131072')
    if not core_mask.is_contiguous():
        core_mask = core_mask.contiguous()
    out = torch.empty((rows, n), dtype=torch.int64, device=core_mask.device)
    _scores[(rows, triton.cdiv(n, block))](seeds, core_mask, ends, out, n, block)
    return out


def random_score_residual(bundle, input_hash, seed, layer, head, *, phase=None,
                          tile_batch=8, block=256):
    """Batch tiles, never partition a tile's candidate set into quota chunks."""
    if tile_batch < 1:
        raise ValueError('tile_batch must be positive')
    phase = phase or (lambda name: nullcontext())
    n = bundle['n']
    k = min(bundle['budget'] - bundle['core_budget'], n)
    if k < 0:
        raise ValueError('core budget exceeds final budget')
    rows = len(bundle['ends'])
    out = torch.full((rows, k), n, dtype=torch.int32, device=bundle['core'].device)
    # For shorter histories assemble() supplies the exact dense branch.
    first = bundle['budget'] // 128
    if n <= bundle['budget'] or not k:
        return out
    for a in range(first, rows, tile_batch):
        b = min(a + tile_batch, rows)
        with phase('random_score_generation'):
            seeds = torch.tensor(seed_values(input_hash, seed, layer, head, range(a, b)),
                                 dtype=torch.int64, device=out.device)
            scores = masked_scores(bundle['mask'][a:b], bundle['ends'][a:b], seeds, block)
        with phase('random_global_topk'):
            indices = scores.topk(k, dim=-1, largest=True, sorted=False).indices
        with phase('residual_index_sort'):
            out[a:b] = indices.to(torch.int32).sort(dim=-1).values
        del scores, seeds, indices
    return out
