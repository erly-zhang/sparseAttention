import math,unittest,torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.selector_grid_cache import SelectorGridLayer,choose,representative


def modes():
    return [f'grid_{s}_{h}_{k}_{p}' for s in ('full','compact') for h,k in [('random','topk'),('w1','topk'),('w16','topk'),('w1','topp'),('w16','topp')] for p in ('protected','none')]

def make(mode,device='cuda'):
    return SelectorGridLayer(3,80 if '_topk_' in mode else 0,0,mode,0.,False,False,top_p=.95,query_window=16 if '_w16_' in mode else 1,eviction_interval=64 if '_compact_' in mode else 1,rep_seed=0,sample_id=19,max_cache_length=256,track_positions=True)

class GridTests(unittest.TestCase):
    def test_selection_contracts(self):
        scores=torch.tensor([.05,.1,.2,.5,.15]);protected=torch.tensor([True,False,False,False,True])
        self.assertEqual(choose(scores,protected,'topk',3,.95).tolist(),[0,3,4])
        self.assertEqual(choose(torch.ones(5),torch.zeros(5,dtype=torch.bool),'topk',2,.95).tolist(),[0,1])
        for p in (.1,.5,.95,1):
            ids=choose(scores,protected,'topp',0,p);extra=ids[~protected[ids]]
            self.assertGreaterEqual(float(scores[extra].sum()/scores[~protected].sum()),p-1e-6)
            if len(extra)>1:self.assertLess(float((scores[extra].sum()-scores[extra].min())/scores[~protected].sum()),p+1e-6)
        self.assertEqual(choose(scores,protected,'topk',9,.95).tolist(),list(range(5)))
        with self.assertRaises(ValueError):choose(scores,protected,'topk',1,.95)
    def test_representative_is_independent_of_global_rng(self):
        before=torch.random.get_rng_state().clone()
        values=[representative(0,19,3,s,32) for s in range(100)]
        self.assertEqual(values,[representative(0,19,3,s,32) for s in range(100)])
        self.assertGreater(len(set(values)),20)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
    def test_window_probability_mean_and_gqa_oracle(self):
        for mode in ('grid_full_w1_topk_none','grid_full_w16_topk_none','grid_compact_w16_topp_protected'):
            l=make(mode,'cpu');l.lazy_initialization(torch.zeros(2,5,2,16))
            g=torch.Generator().manual_seed(831);n=100;l._lengths=[n,n-9];l.cache_seqlens[:]=torch.tensor(l._lengths)
            l.cumulative_length=128;l._num_query_heads=8;l._q_valid=l.window
            l.queries=torch.randn(2,8,l.window,16,generator=g)*3
            l.k_cache.normal_(generator=g)
            for b,length in enumerate(l._lengths):l.slot_positions[b,:length]=torch.arange(length)
            ids,counts=l.select_indices();rep=l._rep_head;kv=rep//4
            for b,length in enumerate(l._lengths):
                per_query=[]
                for t in range(l.window):
                    logits=torch.stack([(l.queries[b,rep,t]*l.k_cache[b,j,kv]).sum() for j in range(length)])/4
                    per_query.append(logits.softmax(-1))
                oracle=torch.stack(per_query).mean(0)
                torch.testing.assert_close(l._last_scores[b,:length],oracle,atol=2e-7,rtol=3e-6)
                protected=torch.zeros(length,dtype=torch.bool)
                if l.protection=='protected':protected=(l.slot_positions[b,:length]<5)|(l.slot_positions[b,:length]>=64)
                expected=choose(oracle,protected,l.selector,l.token_budget,l.top_p)
                self.assertTrue(torch.equal(ids[b,:counts[b]],expected))

    def test_factory(self):
        for mode in modes():
            c=EvictCache(Qwen3Config(num_hidden_layers=2),80 if '_topk_' in mode else 0,0,mode,rkv_lambda=0.,smooth=False,verbose=False,top_p=.95,query_window=16 if '_w16_' in mode else 1,eviction_interval=64 if '_compact_' in mode else 1)
            self.assertTrue(all(type(l) is SelectorGridLayer for l in c.layers))
    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_all_modes_attention_gather_gqa_protection_and_shrink(self):
        for mode in modes():
            layer=make(mode);gen=torch.Generator(device='cuda').manual_seed(83)
            history_k=[];history_v=[];pairs=0;causal=0;seen_events=0
            for t in range(150):
                b=2 if t<145 else 1;n=5 if t==0 else 1
                q=torch.randn(b,n,8,16,device='cuda',dtype=torch.bfloat16,generator=gen)
                k=torch.randn(b,n,2,16,device='cuda',dtype=q.dtype,generator=gen);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=gen)
                old_lengths=list(layer._lengths)
                previous_k=layer.k_cache.clone() if t else None;previous_v=layer.v_cache.clone() if t else None
                layer.update_queries(q.transpose(1,2));before=torch.cuda.get_rng_state().clone()
                out=layer.attention(q,k,v,1/math.sqrt(16))
                self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()),mode)
                expected=[]
                for row in range(b):
                    if t==0:ks=k[row];vs=v[row]
                    elif layer.storage=='full':
                        count=layer._last_selected_ids.shape[1]
                        # Top-p rows can differ: trace stores true positions/count.
                        count=layer.event_trace[-1]['counts'][row]
                        ids=layer._last_selected_ids[row,:count]
                        ks=layer.k_cache[row,ids];vs=layer.v_cache[row,ids]
                    else:
                        ks=torch.cat((previous_k[row,:old_lengths[row]],k[row]),0)
                        vs=torch.cat((previous_v[row,:old_lengths[row]],v[row]),0)
                    logits=torch.einsum('qhd,khd->hqk',q[row].float(),ks.repeat_interleave(4,dim=1).float())/math.sqrt(16)
                    if t==0:logits.masked_fill_(torch.triu(torch.ones(n,n,device='cuda',dtype=torch.bool),1)[None],-float('inf'))
                    ref=torch.einsum('hqk,khd->qhd',logits.softmax(-1),vs.repeat_interleave(4,dim=1).float())
                    expected.append(ref)
                    pairs+=8*(n*(n+1)//2 if t==0 else len(ks))
                torch.testing.assert_close(out.float(),torch.stack(expected),atol=.018,rtol=.035,msg=mode)
                causal+=8*b*(n*(n+1)//2 if t==0 else t+5)
                self.assertEqual(layer.selected_pairs,pairs);self.assertEqual(layer.causal_pairs,causal)
                self.assertEqual(layer.cumulative_length,t+5)
                if layer.storage=='full':self.assertTrue(all(x==t+5 for x in layer._lengths))
                if layer.selection_events>seen_events:
                    event=layer.event_trace[-1]
                    for row,pos in enumerate(event['positions']):
                        self.assertEqual(pos,sorted(set(pos)));self.assertTrue(all(0<=x<t+5 for x in pos))
                        if layer.protection=='protected':
                            required=set(range(5))|set(range(max(5,t+5-64),t+5))
                            self.assertTrue(required<=set(pos),mode)
                        if layer.selector=='topk':self.assertEqual(len(pos),min(80,t+5))
                        if layer.storage=='compact':
                            ids=layer._last_selected_ids[row,:len(pos)]
                            before_k=torch.cat((previous_k[row,:old_lengths[row]],k[row]),0)
                            before_v=torch.cat((previous_v[row,:old_lengths[row]],v[row]),0)
                            self.assertTrue(torch.equal(layer.k_cache[row,:len(pos)],before_k[ids]),mode)
                            self.assertTrue(torch.equal(layer.v_cache[row,:len(pos)],before_v[ids]),mode)
                    seen_events=layer.selection_events
                if t==144:layer.batch_select_indices(torch.tensor([1],device='cuda'));self.assertEqual(layer._batch_ids,[1])
    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_random_frequency_and_replay(self):
        layer=make('grid_full_random_topk_none');layer.token_budget=3
        layer.lazy_initialization(torch.zeros(1,2,2,16,device='cuda',dtype=torch.bfloat16))
        layer._num_query_heads=8;layer._lengths=[10];layer.cache_seqlens.fill_(10);layer.slot_positions[0,:10]=torch.arange(10,device='cuda')
        counts=torch.zeros(10,device='cuda')
        for step in range(2000):
            layer.cumulative_length=step+10;ids,_=layer.select_indices();counts[ids[0]]+=1
            if step==13:
                saved=ids.clone();again,_=layer.select_indices();self.assertTrue(torch.equal(saved,again))
        self.assertTrue(bool(((counts/2000-.3).abs()<.05).all()),counts)

if __name__=='__main__':unittest.main(verbosity=2)
