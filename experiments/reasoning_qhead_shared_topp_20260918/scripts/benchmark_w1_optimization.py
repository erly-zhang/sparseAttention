"""Same-GPU microbench: original, conditional-Q only, conditional-Q + pingpong."""
import json,statistics,sys,time,torch
from pathlib import Path
from kvcompress.engine.representative_compact_recent_cache import RepresentativeCompactRecentLayer as Base
from kvcompress.engine.representative_compact_optimized_cache import RepresentativeCompactOptimizedLayer as Opt
class QOnly(Base):
 def update_queries(self,q):
  if self.cumulative_length and self._cached_tokens+q.shape[2]==self.capacity:Base.update_queries(self,q)
out=Path(sys.argv[1]);torch.set_num_threads(4);torch.cuda.set_device(0)
B=2;P=256;D=128;H=8;HQ=32;dtype=torch.bfloat16;g=torch.Generator(device='cuda').manual_seed(711)
q=torch.randn(B,HQ,1,D,device='cuda',dtype=dtype,generator=g);k=torch.randn(B,1,H,D,device='cuda',dtype=dtype,generator=g);v=torch.randn(k.shape,device='cuda',dtype=dtype,generator=g)
pq=torch.randn(B,P,HQ,D,device='cuda',dtype=dtype,generator=g);pk=torch.randn(B,P,H,D,device='cuda',dtype=dtype,generator=g);pv=torch.randn(pk.shape,device='cuda',dtype=dtype,generator=g)
variants={}
for name,cls in [('original',Base),('conditional_query',QOnly),('optimized',Opt)]:
 l=cls(0,0,0,'qhead_shared_representative_compact_recent_gpu',0.,False,False,reasoning_budget=1024,recent_window=64,eviction_interval=64,sample_id=19)
 l.update_queries(pq.transpose(1,2));l.attention(pq,pk,pv,D**-.5)
 for i in range(1536):l.update_queries(q);l.attention(q.transpose(1,2),k,v,D**-.5)
 variants[name]=l
results={name:[] for name in variants}
for rep in range(6):
 names=list(variants) if rep%2==0 else list(reversed(variants))
 for name in names:
  l=variants[name];torch.cuda.synchronize();start=time.perf_counter()
  for i in range(512):l.update_queries(q);l.attention(q.transpose(1,2),k,v,D**-.5)
  torch.cuda.synchronize();results[name].append(1000*(time.perf_counter()-start)/512)
summary=dict(gpu=torch.cuda.get_device_name(),prompt=P,batch=B,decode_steps_per_trial=512,trials=results,median_ms={name:statistics.median(v) for name,v in results.items()},scope='single layer cache path; excludes model projections/MLP; not end-to-end speedup')
(out/'microbenchmark.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
