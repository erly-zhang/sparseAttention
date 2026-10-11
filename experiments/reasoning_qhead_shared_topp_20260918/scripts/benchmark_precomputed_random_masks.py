"""Selection-only timings; preparation is included. Not model throughput."""
import json,statistics,sys,time
from pathlib import Path
import torch
from transformers import Qwen3Config
from kvcompress.engine.cache_utils import EvictCache
OUT=Path(sys.argv[1]);P=71;B=2;L=36

def make(mode):
 kw=dict(rkv_lambda=0.,smooth=False,verbose=False,sample_id=19,mask_seed=42,recent_window=64)
 if 'compact' in mode:kw.update(reasoning_budget=1024,eviction_interval=64)
 else:kw.update(random_keep_ratio=.4)
 c=EvictCache(Qwen3Config(num_hidden_layers=L),0,0,mode,**kw)
 for layer in c.layers:
  layer.device=torch.device('cuda');layer.prompt_len=P;layer._batch_ids=[0,1];layer._num_query_heads=32
  layer.k_cache=torch.empty(B,0,8,128,device='cuda');layer.capacity=P+1088;layer._cached_tokens=layer.capacity
 return c

def full(mode,start,nsteps=16):
 c=make(mode)
 if hasattr(c,'mask_plan'):c.mask_plan.max_step=start+nsteps-1
 torch.cuda.synchronize();begin=time.perf_counter()
 for t in range(start,start+nsteps):
  for layer in c.layers:
   layer.cumulative_length=P+t;layer.selected_indices_tensor()
 torch.cuda.synchronize();elapsed=time.perf_counter()-begin
 return dict(seconds=elapsed,plan_bytes=c.mask_plan.max_plan_bytes if hasattr(c,'mask_plan') else 0)

def compact(mode):
 c=make(mode);steps=range(1088,32768-P,64)
 torch.cuda.synchronize();begin=time.perf_counter()
 if hasattr(c,'compact_mask_plan'):c.compact_mask_plan.precompute_all(P,32768-P-1,B,'cuda')
 torch.cuda.synchronize();prepared=time.perf_counter()-begin
 for t in steps:
  for layer in c.layers:
   layer.cumulative_length=P+t;layer.select_indices()
 torch.cuda.synchronize();elapsed=time.perf_counter()-begin
 return dict(seconds=elapsed,preparation_seconds=prepared,plan_bytes=c.compact_mask_plan.ids.numel()*8 if hasattr(c,'compact_mask_plan') else 0)
full_old='qhead_shared_mask_random_recent_gpu';full_new='qhead_shared_mask_random_precomputed_recent_gpu'
compact_old='qhead_shared_random_compact_recent_gpu';compact_new='qhead_shared_random_compact_precomputed_gpu'
# Warm both paths before measurement.
full(full_old,1024,2);full(full_new,1024,2)
results=[]
for start in (1024,8192,32000):
 values={full_old:[],full_new:[]}
 for order in ((full_old,full_new),(full_new,full_old),(full_old,full_new)):
  for mode in order:values[mode].append(full(mode,start))
 a=statistics.median(x['seconds'] for x in values[full_old]);b=statistics.median(x['seconds'] for x in values[full_new])
 results.append(dict(kind='full40',start=start,steps=16,baseline_median_seconds=a,precomputed_median_seconds=b,ratio=a/b,reduction=1-b/a,measurements=values))
 print(json.dumps(results[-1]),flush=True)
values={compact_old:[],compact_new:[]}
for order in ((compact_old,compact_new),(compact_new,compact_old),(compact_old,compact_new)):
 for mode in order:values[mode].append(compact(mode))
a=statistics.median(x['seconds'] for x in values[compact_old]);b=statistics.median(x['seconds'] for x in values[compact_new])
results.append(dict(kind='compact_all494events',baseline_median_seconds=a,precomputed_median_seconds=b,ratio=a/b,reduction=1-b/a,measurements=values))
print(json.dumps(results[-1]),flush=True)
OUT.write_text(json.dumps(dict(scope='Selection only, all preparation included; not end-to-end model speed',layers=L,rollouts=B,prompt_tokens=P,results=results),indent=2)+'\n')
