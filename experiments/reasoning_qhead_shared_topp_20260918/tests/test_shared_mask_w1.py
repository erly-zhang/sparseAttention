"""W1: only current post-RoPE query selects over the entire legal KV history."""
import math,os,unittest
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.shared_mask_cache import SharedMaskLayer
from test_shared_mask_cache import step,reference
os.environ['QHEAD_DEBUG']='1';os.environ.pop('QHEAD_LOG',None)
torch.set_num_threads(2)
def layer(**kw):return SharedMaskLayer(0,0,0,'qhead_shared_mask_topp',0.,False,False,query_window=1,top_p=.95,sample_id=17,**kw)

class W1Tests(unittest.TestCase):
 def test_factory_current_query_replaces_prefill_and_previous_queries(self):
  cache=EvictCache(Qwen3Config(num_hidden_layers=2),0,0,'qhead_shared_mask_topp',rkv_lambda=0.,smooth=False,verbose=False,query_window=1)
  self.assertTrue(all(x.query_window==1 for x in cache.layers))
  l=layer();l.update_queries(torch.arange(100.).reshape(1,1,100,1).expand(2,4,100,8))
  self.assertEqual(l.queries.shape,(2,4,1,8));self.assertEqual(l._q_valid,1)
  self.assertTrue(bool((l.queries==99).all()))
  for t in range(17):
   q=torch.full((2,4,1,8),float(-t));l.update_queries(q)
   torch.testing.assert_close(l.queries,q,rtol=0,atol=0)
   self.assertEqual(l._q_valid,1);self.assertEqual(l._q_write_pos,0)

 def test_selector_matches_current_query_conditional_topp_and_can_reselect_history(self):
  l=layer();z=torch.zeros(2,7,2,8);l.prepare(z,z)
  l.cache_seqlens.fill_(7);l.cumulative_length=7;l.prompt_len=2
  # No ties near the Top-p boundary. Old prompt queries favor the opposite key.
  weights=torch.tensor([[.3,.2,.4,.07,.02,.008,.002],[.01,.01,.45,.3,.15,.06,.02]])
  l.k_cache[:,:7,:,0]=weights.log()[:,:,None];l.k_cache[:,7:]=float('nan')
  l.update_queries(torch.full((2,4,19,8),-100.))
  current=torch.zeros(2,4,1,8);current[:,:,:,0]=math.sqrt(8);l.update_queries(current)
  original=l.k_cache[:,:7].clone();rng=torch.random.get_rng_state().clone();ids=l.select_indices()
  for b in range(2):
   p=weights[b,2:].double();p/=p.sum();order=sorted(range(5),key=lambda i:-float(p[i]));mass=0.;chosen=[]
   for j in order:
    chosen.append(j+2);mass+=float(p[j])
    if mass>=.95:break
   self.assertEqual(ids[b].tolist(),sorted([0,1]+chosen))
  self.assertIn(2,ids[0].tolist());self.assertNotIn(6,ids[0].tolist())
  l.update_queries(-current);second=l.select_indices();self.assertIn(6,second[0].tolist())
  torch.testing.assert_close(l.k_cache[:,:7],original,rtol=0,atol=0)
  torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)

 @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
 def test_cuda_current_query_attention_full_history_and_batch_filter(self):
  l=layer();rng=torch.Generator(device='cuda').manual_seed(1821);ks=[];vs=[];pairs=0;causal=0
  for t in range(16):
   n=4 if t==0 else 1
   q=torch.randn(2,n,4,8,device='cuda',dtype=torch.bfloat16,generator=rng)
   k=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=rng)
   v=torch.randn(2,n,2,8,device='cuda',dtype=torch.bfloat16,generator=rng)
   out=step(l,q,k,v);ks.append(k);vs.append(v);fk=torch.cat(ks,1);fv=torch.cat(vs,1);length=fk.shape[1]
   torch.testing.assert_close(l.queries,q[:,-1:].transpose(1,2),rtol=0,atol=0)
   self.assertEqual(l._q_valid,1);self.assertEqual(l.cache_seqlens.tolist(),[length]*2)
   torch.testing.assert_close(l.k_cache[:,:length],fk,rtol=0,atol=0);torch.testing.assert_close(l.v_cache[:,:length],fv,rtol=0,atol=0)
   if t==0:
    pairs+=2*4*n*(n+1)//2;causal=pairs
    for j in range(n):torch.testing.assert_close(out[:,j:j+1].float(),reference(q[:,j:j+1],fk,fv,[list(range(j+1))]*2),rtol=.02,atol=.012)
   else:
    ids=[x.tolist() for x in l._last_selected_ids]
    torch.testing.assert_close(out.float(),reference(q,fk,fv,ids),rtol=.02,atol=.012)
    pairs+=4*sum(map(len,ids));causal+=2*4*length
  self.assertEqual(l.selected_pairs,pairs);self.assertEqual(l.causal_pairs,causal)
  l.batch_select_indices(torch.tensor([1],device='cuda'));self.assertEqual(l._batch_ids,[1])
  q=q[1:];k=k[1:];v=v[1:];out=step(l,q,k,v)
  torch.testing.assert_close(l.queries,q.transpose(1,2),rtol=0,atol=0)
  torch.testing.assert_close(out.float(),reference(q,torch.cat((fk[1:],k),1),torch.cat((fv[1:],v),1),[l._last_selected_ids[0].tolist()]),rtol=.02,atol=.012)

 @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
 def test_w1_does_not_limit_key_history_to_one_or_1024(self):
  l=layer();q=torch.ones(2,1,4,8,device='cuda',dtype=torch.bfloat16);k=torch.ones(2,1,2,8,device='cuda',dtype=torch.bfloat16)
  l.prepare(torch.zeros(2,1100,2,8,device='cuda',dtype=torch.bfloat16),k)
  l.cache_seqlens.fill_(1100);l.cumulative_length=1100;l.prompt_len=2
  old=l.k_cache[:,:1100].clone();step(l,q,k,k)
  self.assertEqual(l.cache_seqlens.tolist(),[1101,1101]);self.assertEqual(l.queries.shape[2],1)
  torch.testing.assert_close(l.k_cache[:,:1100],old,rtol=0,atol=0)
  tiny=layer();step(tiny,q,k,k);step(tiny,q,k,k);self.assertEqual(tiny.cache_seqlens.tolist(),[2,2])
if __name__=='__main__':unittest.main(verbosity=2)
