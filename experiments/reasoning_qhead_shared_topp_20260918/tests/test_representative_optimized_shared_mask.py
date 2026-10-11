"""Bitwise equivalence to the completed W1 implementation, including its RNG."""
import unittest,torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.representative_compact_recent_cache import RepresentativeCompactRecentLayer as Base
from kvcompress.engine.representative_compact_optimized_cache import RepresentativeCompactOptimizedLayer as Opt
from test_shared_mask_cache import step
OLD='qhead_shared_representative_compact_recent_gpu';NEW='qhead_shared_representative_compact_optimized_gpu'
def make(cls,device='cpu',p=5,budget=8,recent=3,interval=4,dtype=torch.float32,hkv=2,dim=8):
 l=cls(3,0,0,NEW if cls is Opt else OLD,0.,False,False,reasoning_budget=budget,recent_window=recent,eviction_interval=interval,query_window=1,rep_seed=0,sample_id=19,track_positions=True)
 l.lazy_initialization(torch.zeros(2,p,hkv,dim,device=device,dtype=dtype));return l
class OptimizedTests(unittest.TestCase):
 def test_factory_and_non_event_query_no_copy(self):
  c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,NEW,rkv_lambda=0.,smooth=False,verbose=False,reasoning_budget=1024,eviction_interval=64,query_window=1)
  self.assertTrue(all(type(l) is Opt for l in c.layers))
  l=make(Opt);q=torch.ones(2,8,1,8);l.update_queries(q);self.assertFalse(hasattr(l,'queries'))
  l.cumulative_length=20;l._cached_tokens=l.capacity-1;l.update_queries(q)
  self.assertTrue(torch.equal(l.queries,q));q.zero_();self.assertTrue(bool(l.queries.any()))
  l._cached_tokens=6;l.update_queries(q);self.assertTrue(bool(l.queries.any()))
 def test_cpu_compaction_exact_ties_protection_and_filter(self):
  for p,budget,recent,interval in [(1,8,0,4),(71,8,3,4),(1300,1024,64,64)]:
   a,b=make(Base,p=p,budget=budget,recent=recent,interval=interval),make(Opt,p=p,budget=budget,recent=recent,interval=interval)
   n=a.capacity;g=torch.Generator().manual_seed(713)
   k=torch.randn(a.k_cache.shape,generator=g);v=torch.randn(k.shape,generator=g)
   for l in (a,b):
    l.k_cache.copy_(k);l.v_cache.copy_(v);l.compact_positions[:]=torch.arange(n);l.cache_seqlens.fill_(n);l._cached_tokens=n;l.cumulative_length=n
    Base.update_queries(l,torch.zeros(2,8,1,8)) # all ties; same event's Q
   state=torch.random.get_rng_state().clone()
   for l in (a,b):l.compact()
   self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
   for round in range(3):
    self.assertTrue(torch.equal(a._last_selected_ids,b._last_selected_ids))
    self.assertTrue(torch.equal(a.compact_positions,b.compact_positions))
    self.assertTrue(torch.equal(a.k_cache[:,:p+budget],b.k_cache[:,:p+budget]))
    self.assertTrue(torch.equal(a.v_cache[:,:p+budget],b.v_cache[:,:p+budget]))
    self.assertTrue(torch.equal(a._rep_gen.get_state(),b._rep_gen.get_state()))
    if round==0:
     for l in (a,b):l.batch_select_indices(torch.tensor([1]))
    kk=torch.randn(a.k_cache[:,p+budget:n].shape,generator=g);vv=torch.randn(kk.shape,generator=g);q=torch.randn(1,8,1,8,generator=g)
    for l in (a,b):
     l.k_cache[:,p+budget:n]=kk;l.v_cache[:,p+budget:n]=vv
     l.compact_positions[:,p+budget:n]=torch.arange(l.cumulative_length,l.cumulative_length+interval)
     l._cached_tokens=n-1;l.cumulative_length+=interval;l.update_queries(q)
     l._cached_tokens=n;l.cache_seqlens.fill_(n);l.compact()
 @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
 def test_cuda_bitwise_attention_masks_scores_rng_and_kv(self):
  for p,budget,recent,interval,steps,hkv,dim,hq in [(5,8,3,4,45,2,8,8),(256,1024,64,64,1285,8,128,32),(1300,8,3,4,29,2,8,8)]:
   a,b=make(Base,'cuda',p,budget,recent,interval,torch.bfloat16,hkv,dim),make(Opt,'cuda',p,budget,recent,interval,torch.bfloat16,hkv,dim)
   g=torch.Generator(device='cuda').manual_seed(914)
   for t in range(steps):
    batch=len(a._batch_ids);n=p if t==0 else 1
    q=torch.randn(batch,n,hq,dim,device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn(batch,n,hkv,dim,device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
    before=a.selection_events;state=torch.cuda.get_rng_state().clone()
    x=step(a,q,k,v);y=step(b,q,k,v)
    self.assertTrue(torch.equal(x,y),(p,t,'attention'))
    self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
    m=a._cached_tokens;self.assertEqual(m,b._cached_tokens)
    self.assertTrue(torch.equal(a.k_cache[:,:m],b.k_cache[:,:m]),(p,t,'K'))
    self.assertTrue(torch.equal(a.v_cache[:,:m],b.v_cache[:,:m]),(p,t,'V'))
    self.assertTrue(torch.equal(a.compact_positions,b.compact_positions))
    if a.selection_events!=before:
     self.assertTrue(torch.equal(a._last_selected_ids,b._last_selected_ids))
     self.assertTrue(torch.equal(a._last_scores,b._last_scores))
     self.assertTrue(torch.equal(a._rep_gen.get_state(),b._rep_gen.get_state()))
    if t==steps//2:
     for l in (a,b):l.batch_select_indices(torch.tensor([1],device='cuda'))
   self.assertEqual((a.selected_pairs,a.causal_pairs,a.selection_events),(b.selected_pairs,b.causal_pairs,b.selection_events))
if __name__=='__main__':unittest.main(verbosity=2)
