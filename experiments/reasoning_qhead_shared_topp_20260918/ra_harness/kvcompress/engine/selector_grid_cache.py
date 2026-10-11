"""Controlled shared selectors: full history or periodic physical KV compaction.

Modes grid_{full,compact,compact500}_{random,w1,w16}_{topk,topp}_{protected,none}.
Dense prefill. Compact selections run AFTER attention every configured interval
(TopK only after cache exceeds K). Full-history selections run BEFORE every
single-query decode attention. All KV heads share positions, never KV values.
Window importance preserves the historical estimator: independent softmax for
each recent post-RoPE Q over ALL currently surviving keys, then probability mean.
No historical-query causal mask in this estimator; actual attention is causal.
Protected Top-p is conditional on unprotected candidates, then union protection.
The opt-in W1 TopK score_mode='qk' ranks raw logits directly; probability mode
remains the default. Actual attention always normalizes the selected scores.
"""
import math
import torch
from kvcompress.engine.cache_utils import EvictLayer, _stable_hash
from kvcompress.engine.random_recent_shared_mask_cache import fused_gather_kv


def decode_mode(mode):
    prefix,storage,scorer,selector,protection=mode.split('_')
    if storage=='compact500':storage='compact'
    if prefix!='grid' or storage not in ('full','compact') or scorer not in ('random','w1','w16') or selector not in ('topk','topp') or protection not in ('protected','none'):
        raise ValueError(mode)
    if scorer=='random' and selector!='topk':raise ValueError('random Top-p is not in the grid')
    return storage,scorer,selector,protection


def representative(seed,sample,layer,step,hq):
    # Private CPU RNG avoids CUDA synchronization and is independent of selector,
    # window, protection, storage, batch shrink and sampling RNG.
    g=torch.Generator().manual_seed(_stable_hash(0x47524550,seed,sample,layer,step))
    return int(torch.randint(hq,(1,),generator=g))


def choose(scores,protected,kind,budget,p):
    """One row oracle-friendly selection; scores are finite, all positions legal."""
    n=scores.numel();fixed=torch.where(protected)[0];eligible=torch.where(~protected)[0]
    if kind=='topk':
        count=min(budget,n)-fixed.numel()
        if count<0:raise ValueError('protection exceeds total K')
        if count>=eligible.numel():return torch.arange(n,device=scores.device)
        # Integer random priority keys are unique; head scores use stable ties.
        if scores.dtype==torch.int64:
            chosen=eligible[scores[eligible].topk(count,sorted=False).indices]
        else:
            chosen=eligible[torch.argsort(scores[eligible],descending=True,stable=True)[:count]]
    else:
        if not eligible.numel():return fixed
        values,order=torch.sort(scores[eligible],descending=True,stable=True)
        total=values.sum()
        if not bool(torch.isfinite(total)):raise RuntimeError('nonfinite score')
        if p==1 or float(total)<=0:chosen=eligible
        else:
            count=min(eligible.numel(),int((values.cumsum(0)<p*total).sum())+1)
            chosen=eligible[order[:count]]
    return torch.cat((fixed,chosen)).sort().values


class SelectorGridLayer(EvictLayer):
    records_attention_metrics=True

    def __init__(self,*args,mask_seed=42,max_cache_length=32768,track_positions=False,score_mode='probability',**kwargs):
        super().__init__(*args,**kwargs)
        self.storage,self.scorer,self.selector,self.protection=decode_mode(self.MODE)
        self.uses_temporary_mask=self.storage=='full'
        self.window=16 if self.scorer=='w16' else 1
        if self.query_window!=self.window:raise ValueError('window/mode mismatch')
        expected_interval=500 if '_compact500_' in self.MODE else (64 if self.storage=='compact' else 1)
        if self.eviction_interval!=expected_interval:raise ValueError('interval/mode mismatch')
        if score_mode not in ('probability','qk'):raise ValueError('unknown score mode')
        if score_mode=='qk' and (self.scorer!='w1' or self.selector!='topk'):raise ValueError('raw QK requires W1 TopK')
        self.score_mode=score_mode
        if score_mode=='qk':
            self.implementation='shared_w1_raw_qk_v1'
            self.rank_backend='stable_sort'
        if self.residual_length or not 0<self.top_p<=1 or self.temperature<=0:raise ValueError('invalid selector config')
        if (self.selector=='topk' and self.token_budget<1) or (self.selector=='topp' and self.token_budget!=0):raise ValueError('Top-p must be uncapped')
        self.mask_seed=mask_seed;self.max_cache_length=max_cache_length
        self.prefill_pairs=0;self.max_selected_tokens=0;self._q_valid=0
        self._lengths=[];self.trace=track_positions;self.event_trace=[]
        self.attention_token_sum=0;self.attention_token_observations=0
        self.selection_token_sum=0;self.selection_token_observations=0
        self._generators={}

    def update_queries(self,q):
        self._num_query_heads=q.shape[1]
        if self.scorer=='random':return
        # Dense prefill does not retain its Q tensor. Decode ring stores <=W Qs.
        if not self.cumulative_length:return
        if q.shape[2]!=1:raise ValueError('single-query decode required')
        step=self.cumulative_length-self.prompt_len+1
        if self.storage=='compact' and (step-1)%self.eviction_interval < self.eviction_interval-self.window:return
        if not hasattr(self,'queries'):
            self.queries=q.new_empty((q.shape[0],q.shape[1],self.window,q.shape[3]))
        slot=(step-1)%self.window
        self.queries[:,:,slot:slot+1].copy_(q)
        self._q_valid=min(self.window,self._q_valid+1)

    def lazy_initialization(self,k,value_states=None):
        b,p,h,d=k.shape;self.prompt_len=p;self.device=k.device;self.dtype=k.dtype
        if self.selector=='topk' and self.protection=='protected' and p+64>self.token_budget:
            raise ValueError('full prompt + recent64 exceeds K; cannot silently clip protection')
        cap=self.max_cache_length
        if self.storage=='compact':
            cap=max(p,self.token_budget)+self.eviction_interval if self.selector=='topk' else min(self.max_cache_length,max(256,p+64))
        self.capacity=cap
        self.k_cache=k.new_empty((b,cap,h,d));self.v_cache=k.new_empty((b,cap,h,d))
        scratch=min(self.token_budget,cap) if self.storage=='full' and self.selector=='topk' else cap
        self._temp_k=k.new_empty((b,scratch,h,d));self._temp_v=torch.empty_like(self._temp_k)
        self.slot_positions=torch.full((b,cap),-1,device=k.device,dtype=torch.long)
        self.cache_seqlens=torch.zeros(b,device=k.device,dtype=torch.int32)
        self._lengths=[0]*b;self._batch_ids=list(range(b));self.is_initialized=True

    def select_indices(self):
        step=self.cumulative_length-self.prompt_len;hq=self._num_query_heads;hkv=self.k_cache.shape[2]
        if hq%hkv:raise ValueError('invalid GQA geometry')
        rep=representative(self.rep_seed,self.sample_id or 0,self.layer_idx,step,hq)
        self._rep_head=rep;self._rep_kv=rep//(hq//hkv)
        extent=max(self._lengths);scores=None
        if self.scorer!='random':
            if self._q_valid<1:raise RuntimeError('missing query history')
            logits=self.queries[:,rep,:self._q_valid]@self.k_cache[:,:extent,self._rep_kv].transpose(1,2)
            if self.score_mode=='probability':logits=logits/(math.sqrt(self.k_cache.shape[-1])*self.temperature)
            valid=torch.arange(extent,device=self.device)[None,:]<self.cache_seqlens[:,None]
            logits.masked_fill_(~valid[:,None,:],-float('inf'))
            scores=logits.squeeze(1) if self.score_mode=='qk' else logits.softmax(-1,dtype=torch.float32).mean(1)
            if self.trace:self._last_scores=scores.clone()
        rows=[]
        for b,n in enumerate(self._lengths):
            positions=self.slot_positions[b,:n]
            protected=torch.zeros(n,device=self.device,dtype=torch.bool)
            if self.protection=='protected':
                protected=(positions<self.prompt_len)|(positions>=max(self.prompt_len,self.cumulative_length-64))
            if self.scorer=='random':
                rollout=self._batch_ids[b]
                if rollout not in self._generators:self._generators[rollout]=torch.Generator(device=self.device)
                rng=self._generators[rollout]
                rng.manual_seed(_stable_hash(0x4752414E,self.mask_seed,self.sample_id or 0,self.layer_idx,step,rollout))
                # random_pp-style independent Uniform float32 scores. Independent
                # secondary random bits resolve finite precision ties; position
                # resolves a double tie. Excluded positions never enter TopK.
                ib=max(1,(n-1).bit_length());sb=min(24,63-24-ib)
                primary=(torch.rand(n,device=self.device,generator=rng)*2**24).long()
                secondary=torch.randint(2**sb,(n,),device=self.device,generator=rng,dtype=torch.long)
                s=(primary<<(sb+ib))|(secondary<<ib)|torch.arange(n,device=self.device)
            else:s=scores[b,:n]
            rows.append(choose(s,protected,self.selector,self.token_budget,self.top_p))
        self.selection_events+=1
        counts=[x.numel() for x in rows]
        self.selection_token_sum+=sum(counts);self.selection_token_observations+=len(counts)
        width=max(counts);ids=torch.zeros((len(rows),width),device=self.device,dtype=torch.long)
        for b,row in enumerate(rows):ids[b,:row.numel()]=row
        self._last_selected_ids=ids
        if self.trace:self.event_trace.append(dict(step=step,representative=rep,counts=counts,positions=[self.slot_positions[b,row].cpu().tolist() for b,row in enumerate(rows)]))
        return ids,counts

    def gather(self,ids):
        if self.device.type=='cuda':fused_gather_kv(self.k_cache,self.v_cache,ids,self._temp_k,self._temp_v)
        else:
            ix=ids[:,:,None,None].expand(-1,-1,*self.k_cache.shape[2:])
            self._temp_k[:,:ids.shape[1]].copy_(self.k_cache.gather(1,ix))
            self._temp_v[:,:ids.shape[1]].copy_(self.v_cache.gather(1,ix))

    def compact(self):
        ids,counts=self.select_indices();self.gather(ids)
        positions=self.slot_positions.gather(1,ids)
        self.k_cache,self._temp_k=self._temp_k,self.k_cache
        self.v_cache,self._temp_v=self._temp_v,self.v_cache
        self.slot_positions.fill_(-1)
        for b,count in enumerate(counts):self.slot_positions[b,:count].copy_(positions[b,:count])
        self._lengths=counts
        self.cache_seqlens.copy_(torch.tensor(counts,device=self.device,dtype=torch.int32))

    def attention(self,q,k,v,scale):
        from flash_attn import flash_attn_with_kvcache
        first=self.cumulative_length==0;b,n,hkv,d=k.shape;hq=q.shape[2]
        if not first and n!=1:raise ValueError('single-token decode required')
        if not self.is_initialized:self.lazy_initialization(k,v)
        required=max(self._lengths)+n
        if required>self.capacity:
            if self.storage!='compact' or self.selector!='topp' or required>self.max_cache_length:
                raise RuntimeError('cache capacity exceeded')
            # Variable Top-p has no fixed K. Grow only when required; preserve
            # surviving chronological KV exactly. Scratch becomes a swap buffer.
            cap=min(self.max_cache_length,max(required,2*self.capacity))
            for name in ('k_cache','v_cache'):
                previous=getattr(self,name);buf=previous.new_empty((b,cap,hkv,d))
                buf[:,:self.capacity].copy_(previous);setattr(self,name,buf)
            self._temp_k=torch.empty_like(self.k_cache);self._temp_v=torch.empty_like(self.v_cache)
            positions=torch.full((b,cap),-1,device=self.device,dtype=torch.long)
            positions[:,:self.capacity].copy_(self.slot_positions);self.slot_positions=positions;self.capacity=cap
        old=self.cumulative_length
        # Compact branch attends to surviving cache + new KV, then evicts.
        if first or self.storage=='compact':
            out=flash_attn_with_kvcache(q,self.k_cache,self.v_cache,k=k,v=v,cache_seqlens=self.cache_seqlens,causal=True,softmax_scale=scale)
        else:
            self.k_cache[:,old:old+n].copy_(k);self.v_cache[:,old:old+n].copy_(v)
        for row,length in enumerate(self._lengths):
            self.slot_positions[row,length:length+n]=torch.arange(old,old+n,device=self.device)
        self._lengths=[x+n for x in self._lengths];self.cache_seqlens+=n;self.cumulative_length+=n
        self.max_visible_tokens=self.cumulative_length
        causal=hq*b*(n*old+n*(n+1)//2)
        if first:
            selected=causal;self.prefill_pairs+=selected;visible=n
        elif self.storage=='full':
            ids,counts=self.select_indices();self.gather(ids)
            lengths=torch.tensor(counts,device=self.device,dtype=torch.int32)
            out=flash_attn_with_kvcache(q,self._temp_k,self._temp_v,cache_seqlens=lengths,causal=True,softmax_scale=scale)
            selected=hq*sum(counts);visible=max(counts)
        else:selected=hq*sum(self._lengths);visible=max(self._lengths)
        self.selected_pairs+=selected;self.causal_pairs+=causal
        self.max_selected_tokens=max(self.max_selected_tokens,visible)
        if not first:
            self.attention_token_sum+=selected//hq;self.attention_token_observations+=b
            step=self.cumulative_length-self.prompt_len
            if self.storage=='compact' and step%self.eviction_interval==0 and (self.selector=='topp' or max(self._lengths)>self.token_budget):self.compact()
        return out

    def batch_select_indices(self,indices):
        ids=indices.tolist()
        self._lengths=[self._lengths[i] for i in ids]
        self._temp_k=self._temp_k[indices].contiguous();self._temp_v=self._temp_v[indices].contiguous()
        self.slot_positions=self.slot_positions[indices].contiguous()
        super().batch_select_indices(indices)
