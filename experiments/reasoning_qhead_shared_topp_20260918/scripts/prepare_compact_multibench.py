import datetime,json,shutil,sys
from pathlib import Path
from compact_multibench_common import sha,rows
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve();data=E/'datasets/compact_multibench_20260926';ref=E/'results/compact_kv_random_reason1024_recent64_20260926'
assert json.loads((data/'dataset_validation.json').read_text())['status']=='passed'
assert json.loads((ref/'collection_status.json').read_text())['phase']=='completed'
for rel,h in json.loads((ref/'inference_source_hashes.json').read_text()).items():assert sha(ref/rel)==h,rel
root.mkdir();(root/'logs').mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(ref/'source'/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in (E/'scripts').glob('*compact_multibench*'):shutil.copyfile(p,root/'source/scripts'/p.name)
for name in ('correctness.log','correctness.json','model_assets_sha256.json','inference_source_hashes.json'):shutil.copyfile(ref/name,root/name)
for name in ('dataset_validation.json','download_manifest.json','gpqa_provenance.json','test_assets_sha256.json'):shutil.copyfile(data/name,root/name)
# Test files are immutable shared links to avoid copying several GB; grader verifies hashes.
counts={'gpqa':198,'aime25':30,'aime26':30,'hmmt':60,'livecodebench':383}
for task,n in counts.items():
 dest=root/'data'/task;dest.mkdir(parents=True);shutil.copyfile(data/task/'test.jsonl',dest/'test.jsonl')
 rr=rows(dest/'test.jsonl');assert len(rr)==n
 if task=='livecodebench':(dest/'livecodebench_tests').symlink_to((data/task/'livecodebench_tests').resolve(),target_is_directory=True)
(root/'inputs').mkdir()
sys.path[:0]=[str(root/'source/ra_harness'),str(root/'source/ra_harness/kvcompress/harness')]
from transformers import AutoTokenizer
from Utils.parser import parse_question
from Utils.data_loader import load_data
import numpy as np
model='/scratch/yl258/jz648/Qwen3-4B';tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
tasks=[];lengths={}
for task,n in counts.items():
 references=[]
 for row in load_data(task,'test',str(root/'data')):
  question=parse_question(row,task)
  if task in ('aime25','aime26','hmmt'):question+='\nPlease reason step by step, and put your final answer within \\boxed{}.'
  tokenized=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
  ids=tokenized['input_ids'][0];assert 0<len(ids)<32768
  references.append(dict(idx=int(row['idx']),input_tokens=len(ids),input_ids_sha256=__import__('hashlib').sha256(ids.numpy().astype('<i8').tobytes()).hexdigest(),prompt_text=tokenizer.decode(ids,skip_special_tokens=True),gold=None if task=='livecodebench' else str(row['answer'])))
 assert {r['idx'] for r in references}==set(range(n))
 (root/'inputs'/f'{task}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in references))
 lengths[task]=dict(min=min(r['input_tokens'] for r in references),max=max(r['input_tokens'] for r in references))
 tasks.append(dict(name=task,count=n,shards={'gpqa':4,'aime25':1,'aime26':1,'hmmt':2,'livecodebench':8}[task]))
cfg=json.loads((ref/'config.json').read_text());cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=tasks,scope='Five task splits across GPQA-D, AIME2025/2026, HMMT2025, LCBv6 medium; compressed method only; two rollouts each',formal_after_all_smoke_validation=True,full_run_authorized=True,smoke_sample_ids=[0,1,2],model=model,input_lengths=lengths,grading='GPQA exact boxed A-D; AIME/HMMT existing math grader; LCB execute all public+private tests with timeout20s/test,2048MiB/process,4 concurrent grading workers',comparison_caveat='Single compressed-method run; no same-input dense or random_pp baseline launched for these datasets.',versions={'gpqa':'official Diamond198; deterministic seed42 choice shuffle','aime25':'AIME2025 I+II30','aime26':'AIME2026 I+II30','hmmt':'HMMT2025 Feb+Nov60','livecodebench':'release_v6 medium383'})
for key in ('dataset_sha256','num_shards','array_tasks','reference_inputs','smoke_source'):cfg.pop(key,None)
cfg.update(dataset='GPQA-D / AIME2025+2026 / HMMT2025 / LCBv6 medium',dataset_hashes={t:sha(root/'data'/t/'test.jsonl') for t in counts},reference_inputs=str(root/'inputs'),implementation_reference=str(ref),auto_full_submission=True,array_tasks=16)
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
work=[dict(task=t['name'],shard=i,num_shards=t['shards']) for t in tasks for i in range(t['shards'])]
(root/'formal_work.json').write_text(json.dumps(work,indent=2)+'\n')
for task in tasks:
 ids=[i for w in work if w['task']==task['name'] for i in range(w['shard'],task['count'],w['num_shards'])]
 assert len(ids)==len(set(ids))==task['count']
files=[p for p in root.rglob('*') if p.is_file() and 'livecodebench_tests' not in p.parts]
(root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(files)},indent=2)+'\n')
print(json.dumps(dict(root=str(root),tasks=tasks,prompt_lengths=lengths,total_questions=sum(counts.values()),total_rollouts=2*sum(counts.values())),indent=2),flush=True)
