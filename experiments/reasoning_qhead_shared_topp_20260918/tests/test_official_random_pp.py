"""Upstream parity and independent numerical/count checks for the official baseline."""
import math,unittest
from pathlib import Path
import hashlib,json
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache,_stable_hash
from kvcompress.engine.official_random_pp_cache import OfficialRandomPPLayer
from kvcompress.engine.official_random_attention.cache_utils import EvictLayer as Upstream
from test_shared_mask_cache import step,reference


def make(layer=0,budget=128,residual=64):
 return OfficialRandomPPLayer(layer,budget,residual,'official_random_pp',0.,False,False,sample_id=13,mask_seed=42)


class OfficialTests(unittest.TestCase):
 def test_vendor_hash_and_factory(self):
  import kvcompress.engine.official_random_attention as pkg
  d=Path(pkg.__file__).parent;p=json.loads((d/'provenance.json').read_text())
  self.assertEqual(p['commit'],'64db9688a12f0926db45fec039370f6ceb1ab4fe')
  for name,h in p['files'].items():self.assertEqual(hashlib.sha256((d/name).read_bytes()).hexdigest(),h)
  c=EvictCache(Qwen3Config(num_hidden_layers=2),1024,64,'official_random_pp',rkv_lambda=0.,smooth=False,verbose=False)
  self.assertTrue(all(isinstance(l,OfficialRandomPPLayer) for l in c.layers))
  self.assertFalse(c.layers[0].uses_temporary_mask)

 def test_upstream_exact_cache_parity_prompt_recent_perhead_rng(self):
  a=make(budget=1024);b=Upstream(0,1024,64,'random_pp',0.,False,False)
  x=torch.arange(1088.).view(1,1088,1,1).expand(2,1088,2,8).clone()
  for l in (a,b):
   l.lazy_initialization(x[:,:70]);l.k_cache.copy_(x);l.v_cache.copy_(x+3)
   l.cache_seqlens.fill_(1088);l.cumulative_length=1088;l.prompt_len=70
  before=torch.random.get_rng_state().clone();a._run_eviction(1088)
  torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(_stable_hash(42,13,0));b._run_eviction(1088)
  torch.testing.assert_close(a.k_cache,b.k_cache,rtol=0,atol=0)
  torch.testing.assert_close(a.v_cache,b.v_cache,rtol=0,atol=0)
  torch.testing.assert_close(a.cache_seqlens,b.cache_seqlens,rtol=0,atol=0)
  for row in range(2):
   for h in range(2):
    ids=a.k_cache[row,:1024,h,0].long().tolist()
    self.assertEqual(ids[:70],list(range(70)))
    self.assertEqual(ids[-64:],list(range(1024,1088)))
    self.assertEqual(ids,sorted(set(ids)))
  self.assertFalse(torch.equal(a.k_cache[0,:960,0],a.k_cache[0,:960,1]))
  self.assertFalse(torch.equal(a.k_cache[0,:960],a.k_cache[1,:960]))
  self.assertEqual(a.get_seq_length(),1088);self.assertEqual(a.cache_seqlens.tolist(),[1024,1024])
  self.assertFalse(hasattr(a,'queries'))

 @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
 def test_cuda_upstream_exact_private_stream(self):
  a=make(budget=1024);b=Upstream(0,1024,64,'random_pp',0.,False,False)
  x=torch.arange(1088.,device='cuda').view(1,1088,1,1).expand(2,1088,2,8).clone()
  for l in (a,b):
   l.lazy_initialization(x[:,:70]);l.k_cache.copy_(x);l.v_cache.copy_(x+3)
   l.cache_seqlens.fill_(1088);l.cumulative_length=1088;l.prompt_len=70
  state=torch.cuda.get_rng_state().clone();cpu=torch.random.get_rng_state().clone()
  a._run_eviction(1088)
  torch.testing.assert_close(torch.cuda.get_rng_state(),state,rtol=0,atol=0)
  torch.testing.assert_close(torch.random.get_rng_state(),cpu,rtol=0,atol=0)
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.cuda.manual_seed(_stable_hash(42,13,0));b._run_eviction(1088)
  torch.testing.assert_close(a.k_cache,b.k_cache,rtol=0,atol=0)
  torch.testing.assert_close(a.v_cache,b.v_cache,rtol=0,atol=0)

 @unittest.skipUnless(torch.cuda.is_available(),'allocated GPU required')
 def test_attention_count_before_eviction_gqa_rope_length_and_batch_filter(self):
  l=make();gen=torch.Generator(device='cuda').manual_seed(3913)
  stored=0;full=0;selected=causal=events=0
  for t in range(245):
   b=2 if t<200 else 1
   if t==200:l.batch_select_indices(torch.tensor([1],device='cuda'))
   n=10 if t==0 else 1
   q=torch.randn(b,n,4,8,device='cuda',dtype=torch.bfloat16,generator=gen)
   k=torch.randn(b,n,2,8,device='cuda',dtype=torch.bfloat16,generator=gen)
   v=torch.randn(k.shape,device='cuda',dtype=k.dtype,generator=gen)
   fk=torch.cat((l.k_cache[:,:stored].clone(),k),1) if t else k
   fv=torch.cat((l.v_cache[:,:stored].clone(),v),1) if t else v
   out=step(l,q,k,v)
   selected+=b*4*(n*stored+n*(n+1)//2);causal+=b*4*(n*full+n*(n+1)//2)
   full+=n;stored+=n
   if t and stored>128 and stored%64==0:stored=128;events+=1
   self.assertEqual(l.cache_seqlens.tolist(),[stored]*b)
   self.assertEqual(l._cached_tokens,stored);self.assertEqual(l.get_seq_length(),full)
   if t:
    torch.testing.assert_close(out.float(),reference(q,fk,fv,[list(range(fk.shape[1]))]*b),rtol=.02,atol=.012)
   else:
    for j in range(n):torch.testing.assert_close(out[:,j:j+1].float(),reference(q[:,j:j+1],fk,fv,[list(range(j+1))]*b),rtol=.02,atol=.012)
  self.assertEqual(l.selected_pairs,selected);self.assertEqual(l.causal_pairs,causal)
  self.assertEqual(l.selection_events,events);self.assertGreater(events,0)
  self.assertEqual(l.max_selected_tokens,192);self.assertEqual(l.prefill_pairs,2*4*55)

if __name__=='__main__':unittest.main(verbosity=2)
