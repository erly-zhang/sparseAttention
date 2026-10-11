"""Bounded single-layer profiling; no generated answers or accuracy claims."""
import json,sys,time,torch
from pathlib import Path
from kvcompress.engine.representative_compact_recent_cache import RepresentativeCompactRecentLayer
out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
torch.set_num_threads(4);assert torch.cuda.is_available()
P=256;B=2;D=128;H=8;HQ=32;steps=1536;device='cuda';dtype=torch.bfloat16
g=torch.Generator(device=device).manual_seed(711)
q=torch.randn(B,HQ,1,D,device=device,dtype=dtype,generator=g)
k=torch.randn(B,1,H,D,device=device,dtype=dtype,generator=g);v=torch.randn(k.shape,device=device,dtype=dtype,generator=g)
pq=torch.randn(B,P,HQ,D,device=device,dtype=dtype,generator=g);pk=torch.randn(B,P,H,D,device=device,dtype=dtype,generator=g);pv=torch.randn(pk.shape,device=device,dtype=dtype,generator=g)
def make():
 l=RepresentativeCompactRecentLayer(0,0,0,'qhead_shared_representative_compact_recent_gpu',0.,False,False,reasoning_budget=1024,recent_window=64,eviction_interval=64,sample_id=19)
 l.update_queries(pq.transpose(1,2));l.attention(pq,pk,pv,D**-.5)
 return l
l=make()
for i in range(steps):l.update_queries(q);l.attention(q.transpose(1,2),k,v,D**-.5)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 for i in range(256):
  with torch.profiler.record_function('save_current_query'):l.update_queries(q)
  with torch.profiler.record_function('cached_attention_and_periodic_compression'):l.attention(q.transpose(1,2),k,v,D**-.5)
 torch.cuda.synchronize()
(out/'baseline_profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=35)+'\n'+prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
measure=[]
for trial in range(3):
 torch.cuda.synchronize();begin=time.perf_counter()
 for i in range(512):l.update_queries(q);l.attention(q.transpose(1,2),k,v,D**-.5)
 torch.cuda.synchronize();measure.append((time.perf_counter()-begin)*1000/512)
(out/'baseline_profile.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),prompt=P,batch=B,steps_profiled=256,wall_ms_per_layer_decode_step=measure),indent=2)+'\n')
print((out/'baseline_profile.json').read_text(),flush=True)
