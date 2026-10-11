"""Independent Bernoulli temporary masks; full KV, expected ratio, no TopK.

A bounded all-layer plan amortizes selection overhead. Different layer/rollout/
step streams are private. Ragged rows are padded only for gather; FlashAttention
receives each row's actual length, so padding never contributes to attention.
"""
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_precomputed_shared_mask_cache import RandomMaskPlan
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer, fused_gather_kv


class BernoulliMaskPlan(RandomMaskPlan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = {}

    def build_step(self, step, prompt_len, rollouts, device):
        if step < 1 or prompt_len < 1 or not rollouts or len(set(rollouts)) != len(rollouts):
            raise ValueError('invalid step, prompt or rollout identities')
        total = self.keep_count(step)
        recent = min(self.recent_window, total)
        older = step - recent
        probability = (total - recent) / older if older else 0.0
        nrows = self.num_layers * len(rollouts)
        # Bernoulli directly produces 0/1 values; no ranking or exact-count repair.
        mask = torch.ones((nrows, prompt_len + step), device=device, dtype=torch.bool)
        for layer in range(self.num_layers):
            for b, rollout in enumerate(rollouts):
                key = (layer, rollout)
                if key not in self.generators:
                    self.generators[key] = torch.Generator(device=device)
                rng = self.generators[key]
                rng.manual_seed(_stable_hash(0x4245524E, self.mask_seed,
                    self.sample_id, step, layer, rollout))
                mask[layer*len(rollouts)+b, prompt_len:prompt_len+older].bernoulli_(probability, generator=rng)
        lengths = mask.sum(-1, dtype=torch.int32)
        # One all-layer transfer per planned step, not one per head/layer.
        counts = lengths.tolist()
        width = max(counts)
        ids = torch.zeros((nrows, width), device=device, dtype=torch.int32)
        rows, positions = mask.nonzero(as_tuple=True)
        ranks = mask.cumsum(-1, dtype=torch.int32)
        ids[rows, ranks[rows, positions].long()-1] = positions.to(torch.int32)
        self.info[step] = (lengths.view(self.num_layers, len(rollouts)),
                           [counts[i:i+len(rollouts)] for i in range(0,nrows,len(rollouts))],
                           total, older*probability*(1-probability))
        return ids.view(self.num_layers, len(rollouts), width)

    def precompute_window(self, *args, **kwargs):
        self.info.clear()
        super().precompute_window(*args, **kwargs)
        extra = sum(x[0].numel()*x[0].element_size() for x in self.info.values())
        self.max_plan_bytes = max(self.max_plan_bytes,
            sum(x.numel()*x.element_size() for x in self.blocks.values()) + extra)

    def bundle(self, layer, step, prompt_len, rollouts, device):
        ids = self.indices(layer, step, prompt_len, rollouts, device)
        lengths, counts, expected, variance = self.info[step]
        slots = [self.slot[r] for r in rollouts]
        if slots == list(range(slots[0], slots[0]+len(slots))):
            lens = lengths[layer,slots[0]:slots[0]+len(slots)]
        else:
            lens = lengths[layer].index_select(0,torch.tensor(slots,device=device))
        return ids, lens, [counts[layer][s] for s in slots], expected, variance


class BernoulliRecentSharedMaskLayer(RandomRecentSharedMaskLayer):
    def __init__(self, *args, mask_precompute_window=16, max_cache_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(mask_precompute_window,int) or not 1 <= mask_precompute_window <= 64:
            raise ValueError('window must be in [1,64]')
        if max_cache_length is not None and (not isinstance(max_cache_length,int) or max_cache_length<1):
            raise ValueError('positive storage bound required')
        self.mask_precompute_window = mask_precompute_window
        self.max_cache_length = max_cache_length
        self.mask_plan = None
        self.decode_selected_tokens = 0
        self.decode_expected_selected_tokens = 0
        self.decode_selection_count_variance = 0.0

    def lazy_initialization(self, key_states, value_states=None):
        if self.max_cache_length is not None and key_states.shape[1] > self.max_cache_length:
            raise ValueError('prompt exceeds storage bound')
        super().lazy_initialization(key_states,value_states)
        if self.max_cache_length is not None:
            shape=(key_states.shape[0],self.max_cache_length,*key_states.shape[2:])
            self.k_cache=key_states.new_empty(shape);self.v_cache=key_states.new_empty(shape)
            if self._track_positions:
                self.positions=self.positions.new_zeros((shape[0],shape[2],self.max_cache_length))

    def prepare(self,key_states,value_states):
        if self.max_cache_length is not None and self.cumulative_length+key_states.shape[1]>self.max_cache_length:
            raise ValueError('append exceeds full-history storage bound')
        return super().prepare(key_states,value_states)

    def attention(self,query,key,value,scale):
        from flash_attn import flash_attn_with_kvcache
        if self.cumulative_length == 0:
            return super().attention(query,key,value,scale)
        if key.shape[1] != 1:
            raise ValueError('single-token decode required')
        kc,vc,_=self.prepare(key,value)
        offset=self.cumulative_length
        kc[:,offset:offset+1].copy_(key);vc[:,offset:offset+1].copy_(value)
        self._advance(1)
        if self.mask_plan is None:raise RuntimeError('cache-owned mask plan required')
        ids,lengths,counts,expected,variance=self.mask_plan.bundle(self.layer_idx,
            self.cumulative_length-self.prompt_len,self.prompt_len,self._batch_ids,self.device)
        b,width=ids.shape
        if self._temp_k is None or width>self._gather_capacity or b>self._gather_batch_capacity:
            self._gather_capacity=max(width,2*self._gather_capacity,256)
            self._gather_batch_capacity=b
            shape=(b,self._gather_capacity,kc.shape[2],kc.shape[3])
            self._temp_k=kc.new_empty(shape);self._temp_v=vc.new_empty(shape)
            self.gather_allocations+=1
        tk=self._temp_k[:b];tv=self._temp_v[:b]
        fused_gather_kv(kc,vc,ids,tk,tv)
        out=flash_attn_with_kvcache(query,tk,tv,cache_seqlens=lengths,
            causal=True,softmax_scale=scale,window_size=(-1,-1))
        self._last_selected_ids=[ids[i,:n] for i,n in enumerate(counts)]
        self.selection_events+=1
        actual=sum(counts);hq=query.shape[2]
        self.selected_pairs+=hq*actual;self.causal_pairs+=hq*b*self.cumulative_length
        self.max_selected_tokens=max(self.max_selected_tokens,max(counts))
        self.decode_selected_tokens+=actual
        self.decode_expected_selected_tokens+=b*(self.prompt_len+expected)
        self.decode_selection_count_variance+=b*variance
        return out
