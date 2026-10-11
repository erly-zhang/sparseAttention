import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]).resolve();cfg=json.loads((root/'config.json').read_text());out=root/'formal';out.mkdir(exist_ok=True)
def dump(name,value):
 (out/name).write_text(json.dumps(value,indent=2)+'\n')
try:
 dump('collection_status.json',dict(status='validating',time=datetime.datetime.now().astimezone().isoformat()))
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(8*1024*1024):h.update(b)
  return h.hexdigest()
 for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
 meta=json.loads((root/'tasks/adeep_s4/meta.json').read_text());inst=torch.load(root/'tasks/adeep_s4/instances.pt',weights_only=True);records={};hardware=[]
 for shard in range(cfg['shards']):
  d=out/f'shard_{shard}';status=json.loads((d/'status.json').read_text());assert status['phase']=='completed'
  assert sha(d/'records.jsonl')==status['records_sha256'];hardware.append(json.loads((d/'hardware.json').read_text()))
  for line in (d/'records.jsonl').open():
   r=json.loads(line);i=r['i'];assert 0<=i<250 and i%cfg['shards']==shard and i not in records
   inp=torch.tensor(meta['stub_ids']+inst[i].tolist(),dtype=torch.int64).numpy().astype('<i8',copy=False).tobytes()
   assert hashlib.sha256(inp).hexdigest()==r['input_ids_sha256'];assert r['target_value']==meta['values'][i]['vx']
   assert r['input_tokens']==4699 and r['stub_tokens']==23 and r['forced_tokens']==4676
   assert len(r['lp_toks'])==len(r['gok'])==4 and abs(sum(r['lp_toks'])-r['lp_sum'])<1e-7
   assert np.isfinite(r['lp_toks']).all() and np.isfinite(r['dense_lp_sum'])
   assert all(r['gok'])==r['fg_ok']==(r['greedy_token_ids']==meta['values'][i]['val_ids'])
   assert r['compression_events_before_value']==len(r['events'])==57
   for j,event in enumerate(r['events']):
    assert event['event']==j+1 and event['cumulative_length']==23+1088+j*64
    assert len(event['box_and_digit_counts_by_layer'])==len(event['representative_q_heads'])==36
    for box,digit in event['box_and_digit_counts_by_layer']:assert 0<=digit<=4 and digit<=box<=16
   records[i]=r
 assert set(records)==set(range(250));rr=[records[i] for i in range(250)]
 smoke=[json.loads(l) for l in (root/'smoke/shard_0/records.jsonl').read_text().splitlines()]
 for r in smoke:
  for k in ['input_ids_sha256','greedy_token_ids','fg_ok','events','coverage_at_question']:assert r[k]==records[r['i']][k],('smoke/formal mismatch',r['i'],k)
  assert abs(r['lp_sum']-records[r['i']]['lp_sum'])<1e-5
 acc=np.array([r['fg_ok'] for r in rr],dtype=float);lp=np.array([r['lp_sum'] for r in rr]);da=np.array([r['dense_fg_ok'] for r in rr],float);dlp=np.array([r['dense_lp_sum'] for r in rr]);cov=np.array([r['coverage_at_question'] for r in rr])
 rng=np.random.default_rng(20260927);ix=rng.integers(0,250,(4000,250))
 def ci(v):return np.percentile(v[ix].mean(1),[2.5,97.5]).tolist()
 report=dict(status='completed',input_alignment='passed/full250',unique_instances=250,model=cfg['model'],method=cfg['method'],correct=int(acc.sum()),retrieval_accuracy=float(acc.mean()),retrieval_bootstrap95=ci(acc),mean_value_logprob=float(lp.mean()),logprob_bootstrap95=ci(lp),dense_correct=int(da.sum()),dense_retrieval_accuracy=float(da.mean()),dense_mean_value_logprob=float(dlp.mean()),actual_compressions_before_query_answer=57,mean_digit_copies_retained_per_layer=float(cov[:,:,1].mean()),fraction_layer_instances_retaining_all4digits=float((cov[:,:,1]==4).mean()),instances_with_zero_digits_in_every_layer=int((cov[:,:,1].sum(1)==0).sum()),mean_diagnostic_forced_wall_seconds=float(np.mean([r['diagnostic_forced_wall_seconds'] for r in rr])),peak_memory_gib=max(r['peak_allocated'] for r in rr)/1024**3,smoke_reproduction='passed/3 identical inputs, outputs, selections, retention traces and LP',dense_parity='passed/3 smoke instances, abs LP difference <0.25 nats',hardware=hardware,scope=cfg['scope'],input_provenance=cfg['input_provenance'],paper_comparison=cfg['paper_comparison'],normalized_recall_R=None,R_note='No keep-everywhere/deleted-everywhere intervention endpoints were run. Dense is an instrument ceiling, not a replacement for those endpoints.',timing_note='Forced-decode diagnostic wall includes retention audit; not a generation speed benchmark or comparable to MATH500 decode times.')
 dump('summary.json',report)
 lines=['# Optimized W1: paper-style planted-passcode probe','',f"Complete250/250 new local instances. Retrieval: {report['correct']}/250 ({report['retrieval_accuracy']:.2%}); mean correct-value log probability: {report['mean_value_logprob']:.4f} nats.",'',f"One-shot dense ceiling: {report['dense_correct']}/250 ({report['dense_retrieval_accuracy']:.2%}), mean log probability {report['dense_mean_value_logprob']:.4f} nats.",'',f"Instance-bootstrap95% CI,4000 resamples: retrieval {report['retrieval_bootstrap95']}; mean log probability {report['logprob_bootstrap95']}.",'',cfg['input_provenance'],'',cfg['paper_comparison'],'','Current W1 unchanged: prompt23 protected outside reasoning1024, recent64 included, one representative per layer/event, correct GQA, stable probability TopK960, chronological shared physical KV compaction every64 steps. Every instance undergoes57 actual compactions before its first answer digit. Four-digit greedy generation is branched before any target token is teacher-forced; exact match agrees with all4 conditional argmax checks.','',f"Mean original digit tokens surviving per layer at query: {report['mean_digit_copies_retained_per_layer']:.3f}/4. Fraction of layer-instance pairs retaining all4: {report['fraction_layer_instances_retaining_all4digits']:.2%}. Instances with zero original digit tokens in every layer: {report['instances_with_zero_digits_in_every_layer']}/250. Retention is diagnostic, not a causal attribution of success/failure.",'',report['R_note'],'',report['timing_note']]
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');dump('collection_status.json',dict(status='completed',time=datetime.datetime.now().astimezone().isoformat()));print(json.dumps(report,indent=2),flush=True)
except Exception as exc:dump('collection_status.json',dict(status='failed',error=repr(exc)));raise
