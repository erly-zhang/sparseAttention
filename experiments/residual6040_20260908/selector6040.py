"""Fixed final budget, shared short core, per-member short residual."""
import contextvars
from contextlib import contextmanager
import torch
import triton
from experiments.probmean_compact_20260906.selector import probability_mean
from experiments.token_compacted_sparse import TokenCompactedIndex,TokenCompactedSelectorStats
from indices import shared_indices,random_residual,scored_residual,assemble,actual_counts
from member_score import member_probability

METHODS={'probmean_shared6144_random4096_allmembers_seed42':'random',
         'probmean_shared6144_member2q4096_allmembers':'member2q'}


class Stats(TokenCompactedSelectorStats):
    def __init__(self):
        super().__init__();self.pending=[];self.expected=[]

    def snapshot(self):
        if self.pending:
            for counts,n,heads in self.pending:
                data=counts.cpu()
                ends=(torch.arange(data.shape[1])+1).mul(128).clamp(max=n)
                expected=ends.clamp(max=10240)[None,:].expand(heads,-1)
                if not torch.equal(data[:,:,0],expected):raise RuntimeError('Actual key budget mismatch')
                self.selected_token_pairs+=int(data[:,:,1].sum())
                self.causal_token_pairs+=heads*n*(n+1)//2
                self.compacted_key_tokens+=int(data[:,:,0].sum())
                self.candidate_key_tokens+=heads*int(ends.sum())
            self.pending=[]
        return super().snapshot()


class Selector:
    def __init__(self,config,mode,*,budget=10240,core=6144):
        self.layers=config['layers'];self.mode=mode;self.budget=budget;self.core=core
        self.current_layer=contextvars.ContextVar('residual6040_layer',default=None)
        self.stats=Stats();self.input_hash='warmup';self.seed=42
        self.profile=False;self.events=[];self.diagnostics=[];self.ranges=()
        self.capture=False;self.captured=[];self.ranked_probability_dump_dir=None
        self.validate=False;self.validation=[]

    @contextmanager
    def phase(self,name):
        if not self.profile:yield;return
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record();self.events.append((name,a,b))

    def __call__(self,q,k,v,block_size,gamma,min_budget,max_budget,tau=0,gqa_interleave=False):
        n,heads=q.shape[1:3];rows=triton.cdiv(n,128);layer=self.current_layer.get()
        width=min(n,self.budget)
        output=torch.empty((heads,rows,width),device=q.device,dtype=torch.int32)
        counts=[];head_order=[]
        for group in self.layers[str(layer)]:
            rep=int(group['representative'])
            kv=lambda h:h%k.shape[2] if gqa_interleave else h//(heads//k.shape[2])
            with self.phase('representative_probability'):
                p=probability_mean(q,k,rep,kv(rep))
            with self.phase('shared_selection_indices'):
                bundle=shared_indices(p,self.budget,self.core)
            for h in group['members']:
                phat=None
                if h==rep or n<=self.budget:residual=None
                elif self.mode=='random':
                    with self.phase('member_sampling'):
                        residual=random_residual(bundle,self.input_hash,self.seed,layer,h)
                else:
                    with self.phase('member_2query_probability'):
                        phat=member_probability(q,k,h,kv(h))
                    with self.phase('member_selection_indices'):
                        residual=scored_residual(phat,bundle)
                with self.phase('final_indices_required_counts'):
                    final=assemble(bundle,residual)
                    output[h].copy_(final)
                    counts.append(actual_counts(final,n));head_order.append(h)
                if self.validate:
                    ends=bundle['ends'][:,None];valid=final<ends
                    ordered=(final[:,1:]>final[:,:-1])|~valid[:,1:]
                    from diagnostics import contains
                    protected=torch.cat((torch.arange(128,device=q.device)[None,:].expand(rows,-1),
                        torch.arange(rows,device=q.device)[:,None]*128+torch.arange(128,device=q.device)[None,:]),1).to(torch.int32)
                    protected_ok=(contains(final,protected,n)|(protected>=ends)).all()
                    self.validation.append(ordered.all()&(final>=0).all()&protected_ok)
                if self.capture:self.captured.append((layer,h,rep,bundle['core'].clone(),final.clone(),bundle['old'].clone(),phat))
                if self.profile:
                    from diagnostics import diagnose
                    with self.phase('extra_diagnostics'):
                        full=p if h==rep else probability_mean(q,k,h,kv(h))
                        data,fields=diagnose(full,phat,bundle,final,h==rep,self.ranges)
                        self.diagnostics.append((layer,h,rep,data,fields))
                del phat,residual,final
            del p,bundle
        self.stats.pending.append((torch.stack(counts),n,heads));self.stats.calls+=1
        starts=torch.arange(heads*rows,device=q.device,dtype=torch.int64).mul(width).view(1,heads,rows)
        lengths=(torch.arange(rows,device=q.device)+1).mul(128).clamp(max=n).clamp(max=self.budget)
        return TokenCompactedIndex(starts,starts+lengths[None,None,:],output.flatten(),
            torch.arange(heads,device=q.device,dtype=torch.int32),heads,rows,128)
