"""Prompt-protected, per-layer uniform random temporary masks; no QK selector.

Each step has a nominal Q-head ID shared across layers. Because selection is
uniform, that ID does not affect the mask. CPU-private RNG streams give identical
choices on different GPUs and survive active-batch filtering. Full K/V and the
actual attention/RoPE/GQA path are inherited unchanged from SharedMaskLayer.
"""
from fractions import Fraction
import json
import os
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.shared_mask_cache import SharedMaskLayer


def keep_count(nonprompt_length, ratio):
    """Exact ceil of decimal ratio * N, avoiding floating-point off-by-one."""
    fraction = Fraction(str(ratio))
    return (nonprompt_length*fraction.numerator + fraction.denominator-1)//fraction.denominator


def nominal_head(rep_seed, sample_id, step, heads):
    rng = torch.Generator(device='cpu').manual_seed(_stable_hash(0x524550, rep_seed, sample_id, step))
    return int(torch.randint(heads, (1,), generator=rng))


def uniform_positions(n, count, seed, sample_id, step, layer, rollout, device):
    if not 0 <= count <= n:
        raise ValueError('invalid random subset size')
    if count == n:
        return torch.arange(n, device=device)
    rng = torch.Generator(device='cpu').manual_seed(_stable_hash(0x4D41534B, seed, sample_id, step, layer, rollout))
    return torch.randperm(n, generator=rng)[:count].sort().values.to(device)


class RandomSharedMaskLayer(SharedMaskLayer):
    def __init__(self, *args, random_keep_ratio, mask_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < random_keep_ratio <= 1:
            raise ValueError('random_keep_ratio must be in (0,1]')
        self.random_keep_ratio = random_keep_ratio
        self.mask_seed = mask_seed
        self.query_window = None
        self.top_p = None

    def update_queries(self, query_states):
        # Retain head count only, never store query vectors or compute scores.
        self._num_query_heads = query_states.shape[1]

    def select_indices(self):
        lengths = self.cache_seqlens.tolist()
        step = self.cumulative_length-self.prompt_len
        sample = self.sample_id if self.sample_id is not None else 0
        self._rep_head = nominal_head(self.rep_seed, sample, step, self._num_query_heads)
        self._rep_kv = self._rep_head // (self._num_query_heads // self.k_cache.shape[2])
        result=[];records=[]
        for b,n in enumerate(lengths):
            pl=self.prompt_len
            if not 0 <= pl <= n <= self.k_cache.shape[1]:
                raise RuntimeError('invalid full-history KV length')
            count=keep_count(n-pl,self.random_keep_ratio)
            chosen=uniform_positions(n-pl,count,self.mask_seed,sample,step,self.layer_idx,self._batch_ids[b],self.device)
            ids=torch.cat((torch.arange(pl,device=self.device),pl+chosen))
            if os.environ.get('QHEAD_DEBUG')=='1':
                assert ids.numel()==pl+count
                assert torch.equal(ids[:pl],torch.arange(pl,device=self.device))
                assert bool((ids[1:]>ids[:-1]).all())
                assert ids.numel()==0 or int(ids[-1])<n
                assert n==self.cumulative_length
                assert not hasattr(self,'queries')
            result.append(ids)
            if os.environ.get('QHEAD_LOG'):
                records.append(dict(implementation='full_kv_random_mask_v1',sample_id=sample,
                    layer=self.layer_idx,step=self.cumulative_length,decode_step=step,rollout=self._batch_ids[b],
                    prompt_len=pl,reasoning_candidates=n-pl,selected_reasoning_K=count,selected_tokens=ids.numel(),
                    full_cache_tokens=n,random_keep_ratio=self.random_keep_ratio,query_window=None,
                    representative_q_head=self._rep_head,representative_role='nominal_only_no_query_scoring'))
        self._last_selected_ids=result
        self.selection_events+=1
        if records:
            with open(os.environ['QHEAD_LOG'],'a') as stream:
                stream.writelines(json.dumps(row)+'\n' for row in records)
        return result
