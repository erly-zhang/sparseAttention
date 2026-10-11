"""Independent W8 mean-probability oracle and event-only query capture parity."""
import math,unittest,torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.representative_window_compact_optimized_cache import RepresentativeWindowCompactOptimizedLayer as Window
from test_shared_mask_cache import step
MODE='qhead_shared_representative_window_compact_optimized_gpu'
class AllQueries(Window):
 def update_queries(self,q):
  self.queries=q[:,:,-self.query_window:].clone() if not hasattr(self,'queries') else torch.cat((self.queries,q),dim=2)[:,:,-self.query_window:].contiguous()
  self._q_valid=self.queries.shape[2]
def make(cls=Window,device='cpu',dtype=torch.float32,p=5,budget=16,recent=4,interval=8,hkv=2,d=8):
 l=cls(3,0,0,MODE,0.,False,False,reasoning_budget=budget,recent_window=recent,eviction_interval=interval,query_window=8,rep_seed=0,sample_id=19,track_positions=True)
 l.lazy_initialization(torch.zeros(2,p,hkv,d,device=device,dtype=dtype));return l
class WindowTests(unittest.TestCase):
 def test_factory_and_invalid_window(self):
  c=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,MODE,rkv_lambda=0.,smooth=False,verbose=False,reasoning_budget=1024,eviction_interval=64,query_window=8)
  self.assertTrue(all(type(l) is Window and l.query_window==8 for l in c.layers))
  with self.assertRaises(ValueError):make(interval=4)
 def test_skip_and_chronological_capture(self):
  l=make();l.cumulative_length=5;l._cached_tokens=l.capacity-9
  l.update_queries(torch.zeros(2,8,1,8));self.assertFalse(hasattr(l,'queries'))
  for i in range(8):
   l._cached_tokens=l.capacity-8+i;l.update_queries(torch.full((2,8,1,8),float(i)))
  self.assertEqual(l._q_valid,8)
  self.assertTrue(torch.equal(l.queries[0,0,:,0],torch.arange(8,dtype=torch.float32)))
  l.batch_select_indices(torch.tensor([1]));self.assertEqual(l.queries.shape[0],1)
  l._cached_tokens=l.capacity-8;l.update_queries(torch.ones(1,8,1,8));self.assertEqual(l._q_valid,1)
  with self.assertRaises(RuntimeError):l.select_indices()
 def test_probability_mean_oracle_ties_and_protection(self):
  for ties in (False,True):
   l=make();n=l.capacity;g=torch.Generator().manual_seed(829)
   l.k_cache.copy_(torch.randn(l.k_cache.shape,generator=g));l.v_cache.copy_(torch.randn(l.v_cache.shape,generator=g))
   l.queries=torch.zeros(2,8,8,8) if ties else torch.randn(2,8,8,8,generator=g)*3
   l._q_valid=8;l._cached_tokens=n;l.cumulative_length=n;l.cache_seqlens.fill_(n);l.compact_positions[:]=torch.arange(n)
   oldk=l.k_cache.clone();oldv=l.v_cache.clone();state=torch.random.get_rng_state().clone();l.compact()
   self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
   rep=l._rep_head;kv=rep//4;self.assertEqual(l._rep_kv,kv)
   # Scalar dot products, independent query softmax and arithmetic mean.
   logits=torch.stack([torch.stack([(l.queries[:,rep,t]*oldk[:,j,kv]).sum(-1) for j in range(n)],dim=1) for t in range(8)],dim=1)/math.sqrt(8)
   scores=logits.softmax(-1).mean(1)
   torch.testing.assert_close(l._last_scores,scores,atol=2e-7,rtol=2e-6)
   if not ties:self.assertGreater((scores-logits.mean(1).softmax(-1)).abs().max().item(),0.001)
   ids=[]
   for b in range(2):
    ranked=sorted(range(l.prompt_len,n-l.recent_window),key=lambda j:(-scores[b,j].item(),j))[:l.reasoning_budget-l.recent_window]
    ids.append(list(range(l.prompt_len))+sorted(ranked)+list(range(n-l.recent_window,n)))
   ids=torch.tensor(ids);self.assertTrue(torch.equal(ids,l._last_selected_ids));self.assertEqual(ids.shape[1],l.prompt_len+l.reasoning_budget)
   ix=ids[:,:,None,None].expand(-1,-1,2,8)
   self.assertTrue(torch.equal(l.k_cache[:,:ids.shape[1]],oldk.gather(1,ix)))
   self.assertTrue(torch.equal(l.v_cache[:,:ids.shape[1]],oldv.gather(1,ix)))
   self.assertTrue(torch.equal(l.compact_positions[:,:ids.shape[1]],ids))
 @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
 def test_cuda_capture_equivalent_to_full_history_ring(self):
  for p,budget,recent,interval,steps,hkv,d,hq in [(5,16,4,8,60,2,8,8),(256,1024,64,64,1220,8,128,32)]:
   a,b=[make(cls,'cuda',torch.bfloat16,p,budget,recent,interval,hkv,d) for cls in (Window,AllQueries)]
   g=torch.Generator(device='cuda').manual_seed(826)
   for t in range(steps):
    bs=len(a._batch_ids);n=p if t==0 else 1
    q=torch.randn(bs,n,hq,d,device='cuda',dtype=torch.bfloat16,generator=g);k=torch.randn(bs,n,hkv,d,device='cuda',dtype=q.dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=q.dtype,generator=g)
    before=a.selection_events;state=torch.cuda.get_rng_state().clone()
    x=step(a,q,k,v);y=step(b,q,k,v)
    self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
    m=a._cached_tokens
    self.assertTrue(torch.equal(a.k_cache[:,:m],b.k_cache[:,:m]));self.assertTrue(torch.equal(a.v_cache[:,:m],b.v_cache[:,:m]))
    self.assertTrue(torch.equal(a.compact_positions,b.compact_positions))
    if a.selection_events!=before:
     self.assertTrue(torch.equal(a.queries,b.queries));self.assertTrue(torch.equal(a._last_scores,b._last_scores));self.assertTrue(torch.equal(a._last_selected_ids,b._last_selected_ids));self.assertTrue(torch.equal(a._rep_gen.get_state(),b._rep_gen.get_state()))
    if t==steps-5:
     for l in (a,b):l.batch_select_indices(torch.tensor([1],device='cuda'))
   self.assertEqual((a.selected_pairs,a.causal_pairs,a.selection_events),(b.selected_pairs,b.causal_pairs,b.selection_events))
if __name__=='__main__':unittest.main(verbosity=2)
