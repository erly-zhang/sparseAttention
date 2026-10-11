"""Full-history KV storage with a temporary representative-head Top-p attention mask.

Prefill is dense. Decode appends full K/V, selects from all valid historical
positions, then attends to a temporary gather. Selection never mutates storage.
"""
import json
import math
import os
import torch
from kvcompress.engine.cache_utils import EvictLayer


class SharedMaskLayer(EvictLayer):
    uses_temporary_mask = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.token_budget != 0 or self.residual_length != 0 or self.eviction_interval != 1:
            raise ValueError('temporary shared masks require budget=0, residual=0, interval=1')
        self.max_selected_tokens = 0
        self.prefill_pairs = 0

    def post_attention_update(self, n_new):
        raise RuntimeError('SharedMaskLayer must use attention(); post-attention eviction is forbidden')

    def _run_eviction(self, cur_len):
        raise RuntimeError('full-history shared masks never evict KV')

    def _run_shared_topp(self, *args):
        raise RuntimeError('full-history shared masks never compact persistent KV')

    def _advance(self, n_new):
        self.cache_seqlens += n_new
        self.cumulative_length += n_new
        if self._track_positions:
            positions = torch.arange(self.cumulative_length-n_new, self.cumulative_length, device=self.device)
            for b, n in enumerate(self.cache_seqlens.tolist()):
                self.positions[b, :, n-n_new:n] = positions
        self.max_visible_tokens = max(self.max_visible_tokens, int(self.cache_seqlens.max()))

    def select_indices(self):
        """Read-only selection over complete KV, including this call's appended key.

        Preserve the existing probability mean over the last <=64 real queries;
        do not change its representative RNG, GQA mapping or sort tie behavior.
        """
        lengths = self.cache_seqlens.tolist()
        Hq = self.queries.shape[1]; Hkv = self.k_cache.shape[2]; D = self.k_cache.shape[3]
        rep = int(torch.randint(Hq, (1,), generator=self._rep_gen, device=self.device).item())
        kv = rep // (Hq // Hkv)
        extent = max(lengths)
        logits = self.queries[:, rep, :self._q_valid] @ self.k_cache[:, :extent, kv].transpose(1, 2)
        logits = logits / (math.sqrt(D) * self.temperature)
        valid = torch.arange(extent, device=self.device)[None, :] < self.cache_seqlens[:, None]
        logits.masked_fill_(~valid[:, None, :], float('-inf'))
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(1)
        result=[]; masses=[]; records=[]
        for b, n in enumerate(lengths):
            pl = self.prompt_len
            if not pl <= n <= self.k_cache.shape[1]:
                raise RuntimeError('invalid full KV length')
            reasoning = scores[b, pl:n]
            total = reasoning.sum()
            if not bool(torch.isfinite(total)):
                raise RuntimeError('non-finite representative-head scores')
            if not reasoning.numel() or float(total) <= 0 or self.top_p == 1:
                ids = torch.arange(n, device=self.device)
                mass = 1.0 if reasoning.numel() else 0.0
            else:
                probabilities = reasoning / total
                sorted_probs, order = probabilities.sort(descending=True)
                count = min(n-pl, int((sorted_probs.cumsum(0) < self.top_p).sum())+1)
                ids = torch.cat((torch.arange(pl, device=self.device), pl+order[:count])).sort().values
                mass = float(probabilities[ids[pl:]-pl].sum())
            if os.environ.get('QHEAD_DEBUG') == '1':
                assert torch.equal(ids[:pl], torch.arange(pl, device=self.device))
                assert bool((ids[1:] > ids[:-1]).all())
                assert int(ids[-1]) < n and (n == pl or mass >= self.top_p-1e-5)
                assert n == self.cumulative_length, 'full history was shortened'
            result.append(ids);masses.append(mass)
            if os.environ.get('QHEAD_LOG'):
                records.append(dict(implementation='full_kv_mask_v1',cache_semantics='full_history_temporary_mask',
                    layer=self.layer_idx,step=self.cumulative_length,batch=b,rollout=self._batch_ids[b],
                    sample_id=self.sample_id,p=self.top_p,query_window=self.query_window,real_query_count=self._q_valid,prompt_len=pl,reasoning_candidates=n-pl,
                    selected_reasoning_K=ids.numel()-pl,selected_reasoning_mass=mass,
                    selected_tokens=ids.numel(),full_cache_tokens=n,total_visible_tokens=n,
                    overall_keep_ratio=ids.numel()/n,reasoning_keep_ratio=(ids.numel()-pl)/(n-pl) if n>pl else 0.,
                    representative_q_head=rep,representative_kv_head=kv,
                    cache_capacity=self.k_cache.shape[1],token_budget=0,eviction_interval=1))
        self._last_selected_ids=result;self._sel_mass=masses
        self._rep_head=rep;self._rep_kv=kv
        self.selection_events += 1
        if records:
            with open(os.environ['QHEAD_LOG'],'a') as stream:
                stream.writelines(json.dumps(row)+'\n' for row in records)
        return result

    def attention(self, query, key, value, scale):
        from flash_attn import flash_attn_with_kvcache
        n_new = key.shape[1]
        first = self.cumulative_length == 0
        if not first and n_new != 1:
            raise ValueError('shared-mask decode requires one token per call')
        kc, vc, lengths = self.prepare(key, value)
        B = key.shape[0]; Hq = query.shape[2]
        if first:
            self.prompt_len = n_new
            output = flash_attn_with_kvcache(query, kc, vc, k=key, v=value,
                cache_seqlens=lengths, causal=True, softmax_scale=scale, window_size=(-1,-1))
            self._advance(n_new)
            pairs = B*Hq*n_new*(n_new+1)//2
            self.selected_pairs += pairs;self.causal_pairs += pairs
            self.prefill_pairs += pairs
            self.max_selected_tokens = n_new
            return output

        # Append exactly once to persistent full-history storage. No gather writes here.
        for b, offset in enumerate(lengths.tolist()):
            kc[b, offset:offset+1].copy_(key[b])
            vc[b, offset:offset+1].copy_(value[b])
        self._advance(1)
        ids = self.select_indices()
        selected_lengths = torch.tensor([i.numel() for i in ids], device=self.device, dtype=torch.int32)
        extent = max(i.numel() for i in ids)
        temp_k = kc.new_zeros((B, extent, kc.shape[2], kc.shape[3]))
        temp_v = vc.new_zeros(temp_k.shape)
        for b, positions in enumerate(ids):
            temp_k[b, :positions.numel()].copy_(kc[b].index_select(0, positions))
            temp_v[b, :positions.numel()].copy_(vc[b].index_select(0, positions))
        # For a single current query all gathered positions are legal causal keys.
        # These temporary buffers never replace k_cache/v_cache or cache_seqlens.
        output = flash_attn_with_kvcache(query, temp_k, temp_v, cache_seqlens=selected_lengths,
            causal=True, softmax_scale=scale, window_size=(-1,-1))
        self.selected_pairs += Hq*sum(i.numel() for i in ids)
        self.causal_pairs += Hq*sum(self.cache_seqlens.tolist())
        self.max_selected_tokens = max(self.max_selected_tokens, extent)
        return output
