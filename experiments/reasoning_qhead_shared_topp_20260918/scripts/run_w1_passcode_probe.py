"""Paper-style forced-decode probe using unchanged optimized W1, batch1."""
import argparse,copy,datetime,gc,hashlib,json,os,platform,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('stage',choices=['smoke','formal']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
root=Path(a.root).resolve();cfg=json.loads((root/'config.json').read_text());sys.path.insert(0,str(root/'source/ra_harness'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
for rel,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
for key in ('KEEPLOG','KEEPLOG_POS','QHEAD_LOG','FORCE_KEEP_RANGE','SYNTH_AUDIT_DIR'):os.environ.pop(key,None)
os.environ['QHEAD_DEBUG']='0'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from kvcompress.engine.modify_forward import wrap_evict_attn_forward
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.official_random_attention.cache_utils import EvictCache as OriginalCache
assert torch.cuda.device_count()==1
gpu=torch.cuda.get_device_properties(0);assert 'A100' in gpu.name and gpu.total_memory>75*1024**3
out=root/a.stage/f'shard_{a.shard}';out.mkdir(parents=True)
def status(phase,**kw):
 d=dict(phase=phase,time=datetime.datetime.now().astimezone().isoformat(),**kw);(out/'status.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
(out/'hardware.json').write_text(json.dumps(dict(node=platform.node(),gpu=gpu.name,memory=gpu.total_memory,job_id=os.environ.get('SLURM_JOB_ID'),batch_size=1),indent=2)+'\n')
try:
 if a.stage=='formal':assert json.loads((root/'smoke/shard_0/status.json').read_text())['phase']=='completed'
 meta=json.loads((root/'tasks/adeep_s4/meta.json').read_text());inst=torch.load(root/'tasks/adeep_s4/instances.pt',weights_only=True);stub=torch.tensor(meta['stub_ids'],device='cuda').view(1,-1)
 NV=meta['n_val_tokens'];Lf=meta['L_forced'];P=stub.shape[1];assert NV==4 and inst.shape==(250,Lf)
 ids=list(range(3)) if a.stage=='smoke' else [i for i in range(cfg['n']) if i%cfg['shards']==a.shard]
 if a.stage=='smoke':
  import subprocess
  status('gpu_regression')
  with (out/'correctness.log').open('x') as f:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'source/tests'),'-p','test*shared*mask*.py','-v'],env={**os.environ,'PYTHONPATH':str(root/'source/ra_harness')},stdout=f,stderr=subprocess.STDOUT,check=True)
 status('loading')
 wrap_evict_attn_forward(cfg['model']);tok=AutoTokenizer.from_pretrained(cfg['model'])
 model=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='flash_attention_2').eval()
 torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 def make_cache(i,dense=False):
  if dense:return OriginalCache(model.config,token_budget=8192,residual_length=64,eviction_mode='recency',rkv_lambda=0.,smooth=False,verbose=False)
  return EvictCache(model.config,token_budget=0,residual_length=0,eviction_mode=cfg['method'],rkv_lambda=0.,smooth=False,verbose=False,query_window=1,reasoning_budget=1024,recent_window=64,eviction_interval=64,rep_seed=0,sample_id=i,track_positions=True,temperature=1.)
 def forward(x,cache,logits_to_keep=1):return model(x,past_key_values=cache,use_cache=True,logits_to_keep=logits_to_keep).logits.float()
 def coverage(cache,digit_positions):
  n=cache.layers[0]._cached_tokens;counts=[]
  for l in cache.layers:
   assert l._cached_tokens==n
   pos=l.compact_positions[:,:n]
   box=((pos>=meta['span_x'][0])&(pos<meta['span_x'][1])).sum()
   dc=(pos[:,:,None]==digit_positions[None,None,:]).any(-1).sum()
   counts.append(torch.stack((box,dc)))
  return torch.stack(counts).cpu().tolist()
 def values_from_logits(initial,cache,target):
  # Explicit greedy continuation, starting BEFORE any correct answer was fed.
  branch=copy.deepcopy(cache);lg=initial;pred=[]
  for k in range(NV):
   nxt=lg.argmax(-1,keepdim=True);pred.append(int(nxt.item()))
   if k<NV-1:lg=forward(nxt,branch)[:,-1]
  del branch;lg=initial;lps=[];gok=[]
  for k in range(NV):
   gold=target[k].view(1,1)
   lps.append(float(torch.log_softmax(lg,dim=-1).gather(-1,gold).item()));gok.append(bool((lg.argmax(-1)==target[k]).item()))
   if k<NV-1:lg=forward(gold,cache)[:,-1]
  assert (pred==target.tolist())==all(gok)
  return dict(lp_toks=lps,lp_sum=sum(lps),gok=gok,fg_ok=all(gok),greedy_token_ids=pred,greedy_value=tok.decode(pred,skip_special_tokens=True))
 with (out/'records.jsonl').open('x') as f:
  for done,i in enumerate(ids):
   status('running',instance=i,completed=done,total=len(ids))
   seq=inst[i].to('cuda');target=seq[-NV:];assert tok.decode(target.tolist())==str(meta['values'][i]['vx'])
   needle=inst[i,meta['span_x'][0]-P:meta['span_x'][1]-P].tolist();v=target.tolist();starts=[j for j in range(len(needle)-NV+1) if needle[j:j+NV]==v];assert len(starts)==1
   digit_positions=torch.arange(meta['span_x'][0]+starts[0],meta['span_x'][0]+starts[0]+NV,device='cuda')
   inp=stub.cpu().view(-1).tolist()+inst[i].tolist();digest=hashlib.sha256(torch.tensor(inp,dtype=torch.int64).numpy().astype('<i8',copy=False).tobytes()).hexdigest()
   # Dense one-shot ceiling on the identical scripted sequence; no answer
   # leakage: logits predicting each digit are taken from its previous position.
   with torch.inference_mode():
    dense=make_cache(i,True);full=torch.tensor(inp,device='cuda').view(1,-1)
    lg=forward(full,dense,NV+1)[0,-NV-1:-1]
    dense_lp=torch.log_softmax(lg,dim=-1).gather(-1,target[:,None]).squeeze(-1).tolist();dense_gok=(lg.argmax(-1)==target).tolist()
    del dense,lg,full
    cache=make_cache(i);torch.cuda.synchronize();t0=time.perf_counter();lg=forward(stub,cache)[:,-1];events=[]
    for t in range(Lf-NV):
     before=cache.layers[0].selection_events;lg=forward(seq[t:t+1].view(1,1),cache)[:,-1]
     if cache.layers[0].selection_events!=before:
      cs=[l.selection_events for l in cache.layers];assert len(set(cs))==1
      c=cache.layers[0].cumulative_length
      assert c==P+1088+(cs[0]-1)*64
      events.append(dict(event=cs[0],cumulative_length=c,box_and_digit_counts_by_layer=coverage(cache,digit_positions),representative_q_heads=[l._rep_head for l in cache.layers]))
    torch.cuda.synchronize();forced_seconds=time.perf_counter()-t0
    assert cache.layers[0].cumulative_length==P+Lf-NV
    expected_events=0 if Lf-NV<1088 else 1+(Lf-NV-1088)//64
    assert len(events)==expected_events
    cv=coverage(cache,digit_positions)
    pred=values_from_logits(lg,cache,target)
    del cache,lg
    parity=None
    if a.stage=='smoke':
     inert=make_cache(i,True);lg=forward(stub,inert)[:,-1]
     for t in range(Lf-NV):lg=forward(seq[t:t+1].view(1,1),inert)[:,-1]
     dense_forced=values_from_logits(lg,inert,target);diff=abs(dense_forced['lp_sum']-sum(dense_lp))
     assert diff<0.25,(i,'teacher-forcing/one-shot dense parity',diff)
     parity=dict(lp_sum=dense_forced['lp_sum'],one_shot_lp_sum=sum(dense_lp),absolute_difference=diff,passed=True,greedy_value=dense_forced['greedy_value'])
     del inert,lg
   rec=dict(i=i,input_ids_sha256=digest,input_tokens=P+Lf,stub_tokens=P,forced_tokens=Lf,target_value=meta['values'][i]['vx'],method=cfg['method'],**pred,dense_lp_sum=sum(dense_lp),dense_lp_toks=dense_lp,dense_fg_ok=all(dense_gok),compression_events_before_value=expected_events,coverage_at_question=cv,events=events,dense_forced_parity=parity,diagnostic_forced_wall_seconds=forced_seconds,peak_allocated=torch.cuda.max_memory_allocated())
   f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps({k:rec[k] for k in ['i','fg_ok','lp_sum','greedy_value','dense_fg_ok','dense_lp_sum','compression_events_before_value']}),flush=True)
   gc.collect();torch.cuda.empty_cache()
 if a.stage=='smoke':
  rr=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()]
  assert all(r['dense_forced_parity']['passed'] for r in rr)
  assert sum(r['dense_fg_ok'] for r in rr)>=2,'Probe ceiling too weak; inspect before full trial'
 status('completed',instances=len(ids),records_sha256=sha(out/'records.jsonl'))
except Exception as exc:status('failed',error=repr(exc));raise
