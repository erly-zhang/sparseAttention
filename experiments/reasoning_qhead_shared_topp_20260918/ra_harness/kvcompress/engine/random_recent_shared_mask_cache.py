"""Per-step shared random masks with budgeted recency and fused reusable KV gather."""
import torch
import triton
import triton.language as tl
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_shared_mask_cache import RandomSharedMaskLayer,keep_count,nominal_head


def random_priority_keys(primary, secondary, positions, index_bits, secondary_bits):
    # Independent primary and secondary random keys, then original position.
    # Position bits guarantee uniqueness even if BOTH random keys collide.
    return ((primary << (secondary_bits+index_bits)) |
            (secondary << index_bits) | positions)


def gpu_uniform_positions(n, count, generator, device):
    if not 0 <= count <= n:
        raise ValueError('invalid sample size')
    if count == 0:
        return torch.empty(0,device=device,dtype=torch.long)
    if count == n:
        return torch.arange(n,device=device)
    ib=max(1,(n-1).bit_length());sb=63-32-ib
    if sb<1:
        raise ValueError('candidate length exceeds supported priority key range')
    # u=primary/2**32 is a finite-precision Uniform[0,1) score.
    primary=torch.randint(2**32,(n,),device=device,dtype=torch.int64,generator=generator)
    secondary=torch.randint(2**sb,(n,),device=device,dtype=torch.int64,generator=generator)
    keys=random_priority_keys(primary,secondary,torch.arange(n,device=device),ib,sb)
    return keys.topk(count,sorted=False).indices.sort().values


@triton.jit(do_not_specialize=["N","KS","VS","TS"])
def _gather_kv(K,V,IDS,TK,TV,N,HD:tl.constexpr,
               KS,VS,TS,BLOCK:tl.constexpr):
    b=tl.program_id(1)
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<N*HD
    pos=tl.load(IDS+b*N+i//HD,mask=mask,other=0)
    inner=i%HD
    kval=tl.load(K+b*KS+pos*HD+inner,mask=mask,other=0)
    vval=tl.load(V+b*VS+pos*HD+inner,mask=mask,other=0)
    tl.store(TK+b*TS+i,kval,mask=mask)
    tl.store(TV+b*TS+i,vval,mask=mask)


def fused_gather_kv(k,v,ids,temp_k,temp_v):
    """Gather K and V directly into reusable buffers in one CUDA kernel."""
    b,n=ids.shape;hd=k.shape[2]*k.shape[3]
    _gather_kv[(triton.cdiv(n*hd,512),b)](k,v,ids,temp_k,temp_v,n,hd,
        k.stride(0),v.stride(0),temp_k.stride(0),BLOCK=512)


class RandomRecentSharedMaskLayer(RandomSharedMaskLayer):
    def __init__(self,*args,recent_window=64,**kwargs):
        super().__init__(*args,**kwargs)
        if recent_window<0:raise ValueError('recent_window must be nonnegative')
        self.recent_window=recent_window
        self._mask_generators={}
        self._gather_capacity=0
        self._gather_batch_capacity=0
        self._temp_k=None;self._temp_v=None;self._selected_lengths=None
        self.gather_allocations=0

    def prepare(self,key_states,value_states):
        if not self.is_initialized:self.lazy_initialization(key_states,value_states)
        # All active rows are rollouts of the same prompt at the same decode step.
        # This is full history, so the authoritative Python length is exact and
        # avoids a GPU->CPU max().item() on every layer and every token.
        required=self.cumulative_length+key_states.shape[1]
        if required>self.k_cache.shape[1]:
            capacity=max(required,2*self.k_cache.shape[1])
            for name in ('k_cache','v_cache'):
                old=getattr(self,name)
                new=old.new_empty((old.shape[0],capacity,*old.shape[2:]))
                new[:,:self.cumulative_length].copy_(old[:,:self.cumulative_length])
                setattr(self,name,new)
            if hasattr(self,'positions'):
                old=self.positions;self.positions=old.new_empty((old.shape[0],old.shape[1],capacity))
                self.positions[:,:,:self.cumulative_length].copy_(old[:,:,:self.cumulative_length])
        return self.k_cache,self.v_cache,self.cache_seqlens

    def _advance(self,n_new):
        self.cache_seqlens+=n_new
        self.cumulative_length+=n_new
        self.max_visible_tokens=max(self.max_visible_tokens,self.cumulative_length)
        if self._track_positions:
            self.positions[:,:,self.cumulative_length-n_new:self.cumulative_length]=torch.arange(
                self.cumulative_length-n_new,self.cumulative_length,device=self.device)

    def reasoning_keep_count(self, reasoning):
        return keep_count(reasoning, self.random_keep_ratio)

    def select_indices(self):
        n=self.cumulative_length;pl=self.prompt_len;reasoning=n-pl
        total=self.reasoning_keep_count(reasoning)
        recent=min(self.recent_window,total)
        older=reasoning-recent;random_count=total-recent
        sample=self.sample_id if self.sample_id is not None else 0
        self._rep_head=nominal_head(self.rep_seed,sample,reasoning,self._num_query_heads)
        self._rep_kv=self._rep_head//(self._num_query_heads//self.k_cache.shape[2])
        prefix=torch.arange(pl,device=self.device)
        tail=torch.arange(n-recent,n,device=self.device)
        result=[]
        for rollout in self._batch_ids:
            rng=self._mask_generators.get(rollout)
            if rng is None:
                rng=torch.Generator(device=self.device);self._mask_generators[rollout]=rng
            rng.manual_seed(_stable_hash(0x475055524543,self.mask_seed,sample,reasoning,self.layer_idx,rollout))
            # Prompt, recent and future positions are excluded before TopK by
            # constructing scores ONLY for the eligible older non-prompt range.
            chosen=gpu_uniform_positions(older,random_count,rng,self.device)
            result.append(torch.cat((prefix,pl+chosen,tail)))
        self._last_selected_ids=result;self.selection_events+=1
        return result

    def selected_indices_tensor(self):
        return torch.stack(self.select_indices())

    def attention(self,query,key,value,scale):
        from flash_attn import flash_attn_with_kvcache
        n_new=key.shape[1];first=self.cumulative_length==0
        if not first and n_new!=1:raise ValueError('single-token decode required')
        kc,vc,lengths=self.prepare(key,value)
        b=key.shape[0];hq=query.shape[2]
        if first:
            self.prompt_len=n_new
            out=flash_attn_with_kvcache(query,kc,vc,k=key,v=value,cache_seqlens=lengths,
                causal=True,softmax_scale=scale,window_size=(-1,-1))
            self._advance(n_new)
            pairs=b*hq*n_new*(n_new+1)//2
            self.selected_pairs+=pairs;self.causal_pairs+=pairs;self.prefill_pairs+=pairs
            self.max_selected_tokens=n_new
            return out
        offset=self.cumulative_length
        kc[:,offset:offset+1].copy_(key);vc[:,offset:offset+1].copy_(value)
        self._advance(1)
        ids=self.selected_indices_tensor()
        selected=ids.shape[1]
        if self._temp_k is None or selected>self._gather_capacity or b>self._gather_batch_capacity:
            self._gather_capacity=max(selected,2*self._gather_capacity,256)
            self._gather_batch_capacity=b
            shape=(b,self._gather_capacity,kc.shape[2],kc.shape[3])
            self._temp_k=kc.new_empty(shape);self._temp_v=vc.new_empty(shape)
            self.gather_allocations+=1
        tk=self._temp_k[:b];tv=self._temp_v[:b]
        fused_gather_kv(kc,vc,ids,tk,tv)
        if self._selected_lengths is None or self._selected_lengths.shape[0]!=b:
            self._selected_lengths=torch.empty(b,device=self.device,dtype=torch.int32)
        self._selected_lengths.fill_(selected)
        out=flash_attn_with_kvcache(query,tk,tv,cache_seqlens=self._selected_lengths,
            causal=True,softmax_scale=scale,window_size=(-1,-1))
        self.selected_pairs+=hq*b*selected;self.causal_pairs+=hq*b*self.cumulative_length
        self.max_selected_tokens=max(self.max_selected_tokens,selected)
        return out
