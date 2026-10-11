"""Exact legacy random masks, generated ahead in bounded all-layer batches.

This plan never consumes Q/K/V values, global RNG or token contents. Layer and
original rollout identities stay in the seed; sharing a plan does NOT share one
mask across layers. All preparation runs inside the timed decode path by default.
"""
from fractions import Fraction
import torch
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.random_shared_mask_cache import nominal_head
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer, random_priority_keys


class RandomMaskPlan:
    def __init__(self, num_layers, ratio, recent_window=64, mask_seed=42,
                 sample_id=0, rep_seed=0, window=16):
        if num_layers < 1 or not isinstance(window, int) or not 1 <= window <= 64:
            raise ValueError('positive layer count and window in [1,64] required')
        if not 0 < ratio <= 1 or recent_window < 0:
            raise ValueError('invalid keep ratio or recent window')
        self.num_layers=num_layers;self.ratio=Fraction(str(ratio))
        self.recent_window=recent_window;self.mask_seed=mask_seed
        self.sample_id=sample_id if sample_id is not None else 0
        self.rep_seed=rep_seed;self.window=window;self.max_step=None
        self.prompt_len=None;self.device=None;self.generators={}
        self.blocks={};self.heads={};self.rollouts=();self.slot={}
        self.refills=0;self.prepared_steps=0;self.prepared_layer_rollout_masks=0
        self.consumed_layer_rollout_masks=0;self.max_plan_bytes=0

    def keep_count(self, step):
        return (step*self.ratio.numerator+self.ratio.denominator-1)//self.ratio.denominator

    def build_step(self, step, prompt_len, rollouts, device):
        """Build [layers, original-active-rollouts, selected] chronological IDs.

        Random tensors use the exact old 1D torch.randint calls and seed sequence.
        Only deterministic key packing, TopK and sorting are batched across rows.
        Position is the unique final tie-break, so row batching cannot change ties.
        """
        if step < 1 or prompt_len < 1 or not rollouts or len(set(rollouts))!=len(rollouts):
            raise ValueError('invalid step, prompt or rollout identities')
        total=self.keep_count(step);recent=min(self.recent_window,total)
        older=step-recent;count=total-recent
        nrows=self.num_layers*len(rollouts)
        if count==0:
            chosen=torch.empty((nrows,0),device=device,dtype=torch.long)
        elif count==older:
            chosen=torch.arange(older,device=device).expand(nrows,-1)
        else:
            ib=max(1,(older-1).bit_length());sb=63-32-ib
            if sb<1:raise ValueError('candidate length exceeds priority-key range')
            primaries=[];secondaries=[]
            for layer in range(self.num_layers):
                for rollout in rollouts:
                    key=(layer,rollout)
                    rng=self.generators.get(key)
                    if rng is None:
                        rng=torch.Generator(device=device);self.generators[key]=rng
                    rng.manual_seed(_stable_hash(0x475055524543,self.mask_seed,
                        self.sample_id,step,layer,rollout))
                    primaries.append(torch.randint(2**32,(older,),device=device,dtype=torch.int64,generator=rng))
                    secondaries.append(torch.randint(2**sb,(older,),device=device,dtype=torch.int64,generator=rng))
            primary=torch.stack(primaries);secondary=torch.stack(secondaries)
            keys=random_priority_keys(primary,secondary,torch.arange(older,device=device),ib,sb)
            chosen=keys.topk(count,dim=-1,sorted=False).indices.sort(dim=-1).values
        prefix=torch.arange(prompt_len,device=device).expand(nrows,-1)
        tail=torch.arange(prompt_len+step-recent,prompt_len+step,device=device).expand(nrows,-1)
        ids=torch.cat((prefix,prompt_len+chosen,tail),dim=-1)
        if prompt_len+step>=2**31:raise ValueError('indices exceed int32 storage')
        return ids.to(torch.int32).view(self.num_layers,len(rollouts),prompt_len+total)

    def precompute_window(self, start, prompt_len, rollouts, device):
        device=torch.device(device);rollouts=tuple(rollouts)
        if self.prompt_len is not None and (prompt_len!=self.prompt_len or device!=self.device):
            raise ValueError('a mask plan is private to one prompt and device')
        self.prompt_len=prompt_len;self.device=device
        end=start+self.window
        if self.max_step is not None:
            if start>self.max_step:raise ValueError('requested step exceeds generation bound')
            end=min(end,self.max_step+1)
        self.blocks.clear();self.heads.clear();self.rollouts=rollouts
        self.slot={r:i for i,r in enumerate(rollouts)}
        for step in range(start,end):
            self.blocks[step]=self.build_step(step,prompt_len,rollouts,device)
        self.refills+=1;self.prepared_steps+=end-start
        self.prepared_layer_rollout_masks+=(end-start)*self.num_layers*len(rollouts)
        self.max_plan_bytes=max(self.max_plan_bytes,sum(x.numel()*x.element_size() for x in self.blocks.values()))

    def indices(self, layer, step, prompt_len, rollouts, device):
        if not 0<=layer<self.num_layers:raise ValueError('invalid layer')
        if step not in self.blocks:
            self.precompute_window(step,prompt_len,rollouts,device)
        if prompt_len!=self.prompt_len or torch.device(device)!=self.device:
            raise ValueError('plan binding changed')
        slots=[self.slot[r] for r in rollouts]
        ids=self.blocks[step][layer]
        # Normal batch2 -> batch1 survivor filtering can use an allocation-free view.
        if slots==list(range(slots[0],slots[0]+len(slots))):
            ids=ids[slots[0]:slots[0]+len(slots)]
        else:
            ids=ids.index_select(0,torch.tensor(slots,device=device))
        self.consumed_layer_rollout_masks+=len(slots)
        return ids

    def nominal_head(self, step, num_heads):
        key=(step,num_heads)
        if key not in self.heads:
            self.heads[key]=nominal_head(self.rep_seed,self.sample_id,step,num_heads)
        return self.heads[key]


class RandomPrecomputedSharedMaskLayer(RandomRecentSharedMaskLayer):
    def __init__(self,*args,mask_precompute_window=16,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(mask_precompute_window,int) or not 1<=mask_precompute_window<=64:
            raise ValueError('mask_precompute_window must be in [1,64]')
        self.mask_precompute_window=mask_precompute_window
        self.mask_plan=None

    def selected_indices_tensor(self):
        if self.mask_plan is None:raise RuntimeError('precomputed layers require a cache-owned mask plan')
        step=self.cumulative_length-self.prompt_len
        ids=self.mask_plan.indices(self.layer_idx,step,self.prompt_len,self._batch_ids,self.device)
        self._rep_head=self.mask_plan.nominal_head(step,self._num_query_heads)
        self._rep_kv=self._rep_head//(self._num_query_heads//self.k_cache.shape[2])
        self._last_selected_ids=list(ids.unbind(0));self.selection_events+=1
        return ids

    def select_indices(self):
        return list(self.selected_indices_tensor().unbind(0))
