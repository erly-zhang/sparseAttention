"""Precompute every possible random compaction mask BEFORE model inference.

Masks refer to the chronological surviving-cache slots at an event, not original
sequence IDs. Events occur at reasoning B+I, B+2I, ... . Their coordinates and
candidate counts are known from prompt length, max_length and the fixed policy.
The legacy private random calls and exact int64 priority tie-break are retained.
"""
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_recent_shared_mask_cache import random_priority_keys
from kvcompress.engine.random_compact_recent_cache import RandomCompactRecentLayer


class CompactRandomMaskPlan:
    def __init__(self,layers,budget,interval,recent=64,seed=42,sample_id=0,chunk_events=16):
        if min(layers,budget,interval,chunk_events)<1 or chunk_events>64 or not 0<=recent<=budget:
            raise ValueError('invalid compact mask-plan configuration')
        self.layers=layers;self.budget=budget;self.interval=interval;self.recent=recent
        self.seed=seed;self.sample_id=sample_id or 0;self.chunk_events=chunk_events
        self.ids=None;self.steps=();self.prompt_len=None;self.rollouts=0
        self.prepared_masks=0;self.consumed_masks=0

    def precompute_all(self,prompt_len,max_decode_steps,batch_size,device):
        if self.ids is not None:raise RuntimeError('plan already prepared; create one plan per generation')
        if prompt_len<1 or max_decode_steps<0 or batch_size<1:raise ValueError('invalid generation shape')
        self.prompt_len=prompt_len;self.rollouts=batch_size
        self.steps=tuple(range(self.budget+self.interval,max_decode_steps+1,self.interval))
        self.ids=torch.empty((len(self.steps),self.layers,batch_size,prompt_len+self.budget),device=device,dtype=torch.long)
        if not self.steps:return
        n=self.budget+self.interval-self.recent;k=self.budget-self.recent
        ib=max(1,(n-1).bit_length());sb=63-32-ib
        if sb<1:raise ValueError('candidate length exceeds legacy priority key range')
        generators={(l,b):torch.Generator(device=device) for l in range(self.layers) for b in range(batch_size)}
        prefix=torch.arange(prompt_len,device=device)
        end=prompt_len+self.budget+self.interval
        tail=torch.arange(end-self.recent,end,device=device)
        for start in range(0,len(self.steps),self.chunk_events):
            steps=self.steps[start:start+self.chunk_events];nrows=len(steps)*self.layers*batch_size
            if k==0:
                chosen=torch.empty((nrows,0),device=device,dtype=torch.long)
            else:
                primaries=[];secondaries=[]
                for step in steps:
                    for layer in range(self.layers):
                        for rollout in range(batch_size):
                            rng=generators[layer,rollout]
                            rng.manual_seed(_stable_hash(0x475055524543,self.seed,self.sample_id,step,layer,rollout))
                            primaries.append(torch.randint(2**32,(n,),device=device,dtype=torch.int64,generator=rng))
                            secondaries.append(torch.randint(2**sb,(n,),device=device,dtype=torch.int64,generator=rng))
                keys=random_priority_keys(torch.stack(primaries),torch.stack(secondaries),torch.arange(n,device=device),ib,sb)
                chosen=keys.topk(k,dim=-1,sorted=False).indices.sort(dim=-1).values
            ids=torch.cat((prefix.expand(nrows,-1),prompt_len+chosen,tail.expand(nrows,-1)),dim=-1)
            self.ids[start:start+len(steps)].copy_(ids.view(len(steps),self.layers,batch_size,prompt_len+self.budget))
        self.prepared_masks=len(self.steps)*self.layers*batch_size

    def indices(self,layer,step,prompt_len,rollouts):
        if self.ids is None:raise RuntimeError('compact masks must be prepared before inference')
        offset=step-self.budget-self.interval
        if offset<0 or offset%self.interval or prompt_len!=self.prompt_len:
            raise ValueError('unexpected compaction event')
        event=offset//self.interval
        if event>=len(self.steps):raise ValueError('event exceeds precomputed generation bound')
        if not 0<=layer<self.layers or not rollouts or any(r<0 or r>=self.rollouts for r in rollouts):
            raise ValueError('invalid layer or original rollout')
        ids=self.ids[event,layer]
        if list(rollouts)==list(range(rollouts[0],rollouts[0]+len(rollouts))):
            ids=ids[rollouts[0]:rollouts[0]+len(rollouts)]
        else:ids=ids.index_select(0,torch.tensor(rollouts,device=ids.device))
        self.consumed_masks+=len(rollouts)
        return ids


class RandomCompactPrecomputedLayer(RandomCompactRecentLayer):
    def __init__(self,*args,mask_precompute_window=16,**kwargs):
        super().__init__(*args,**kwargs)
        self.mask_precompute_window=mask_precompute_window
        self.compact_mask_plan=None

    def select_indices(self):
        if self._cached_tokens!=self.capacity:raise RuntimeError('selection is only valid at compaction')
        if self.compact_mask_plan is None:raise RuntimeError('missing cache-owned compaction plan')
        return self.compact_mask_plan.indices(self.layer_idx,self.cumulative_length-self.prompt_len,
            self.prompt_len,self._batch_ids)
