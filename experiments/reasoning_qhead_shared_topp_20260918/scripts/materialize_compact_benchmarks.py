import base64,csv,hashlib,io,json,pickle,random,re,zlib
from pathlib import Path
E=Path(__file__).resolve().parents[1];root=E/'datasets/compact_multibench_20260926'
class DataUnpickler(pickle.Unpickler):
 def find_class(self,module,name):raise ValueError('Unexpected executable pickle object')
# Use the authorized official GPQA Diamond CSV, not fixed-correct-option upstream conversion.
original=list(csv.DictReader((root/'raw/gpqa_diamond.csv').open()));assert len(original)==198
up=root/'gpqa/test.jsonl';backup=root/'raw/gpqa_vase_fixed_A.jsonl'
if not backup.exists():backup.write_bytes(up.read_bytes())
rows=[];answer_counts={c:0 for c in 'ABCD'}
for idx,r in enumerate(original):
 choices=[r['Correct Answer'],r['Incorrect Answer 1'],r['Incorrect Answer 2'],r['Incorrect Answer 3']]
 identity=r['Record ID'];seed=int.from_bytes(hashlib.sha256(('42:'+identity).encode()).digest()[:8],'little')
 order=list(range(4));random.Random(seed).shuffle(order);answer='ABCD'[order.index(0)];answer_counts[answer]+=1
 question=r['Question'].strip()+'\n\n'+'\n'.join(f'{"ABCD"[j]}. {choices[k].strip()}' for j,k in enumerate(order))+'\n\nPlease reason step by step, and put the letter of your final answer within \\boxed{}.'
 rows.append(dict(idx=idx,id=identity,question=question,answer=answer,choice_order=order))
up.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
print('GPQA official198, deterministic choice distribution',answer_counts,flush=True)
# Map pinned release_v6 global indices to the exact project medium subset.
problems=[json.loads(x) for x in (root/'livecodebench/test.jsonl').open()];wanted={r['global_id']:r for r in problems};assert len(wanted)==383
out=root/'livecodebench/livecodebench_tests';out.mkdir(exist_ok=True);count=0;gid=0;manifest={};test_counts=[]
for i in range(1,7):
 path=root/'raw/lcb'/('test.jsonl' if i==1 else f'test{i}.jsonl')
 with path.open() as f:
  for line in f:
   row=json.loads(line)
   if gid in wanted:
    ref=wanted[gid];assert str(row['question_id'])==str(ref['question_id']) and str(row['contest_id'])==str(ref['contest_id'])
    assert row['difficulty']=='medium'
    pub=json.loads(row['public_test_cases'])
    try:priv=json.loads(row['private_test_cases'])
    except json.JSONDecodeError:
     raw=DataUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(row['private_test_cases'])))).load();priv=json.loads(raw) if isinstance(raw,(str,bytes)) else raw
    cases=pub+priv;assert len(cases)>0
    payload={'input_output':json.dumps({'inputs':[x['input'] for x in cases],'outputs':[x['output'] for x in cases],'fn_name':json.loads(row['metadata']).get('func_name')})}
    data=json.dumps(payload).encode();assert hashlib.md5(data).hexdigest()==ref['tests']['md5'],('LCB tests changed',gid)
    dest=out/ref['tests']['fname'];assert dest.parent==out;dest.write_bytes(data)
    manifest[dest.name]=hashlib.sha256(data).hexdigest();count+=1;test_counts.append(len(cases))
   gid+=1
 print('LCB raw shard',i,'processed, tests materialized',count,flush=True)
assert gid==1055 and count==383
(root/'test_assets_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'dataset_validation.json').write_text(json.dumps(dict(status='passed',counts={'gpqa':198,'aime25':30,'aime26':30,'hmmt':60,'livecodebench':383},gpqa_source='authorized official gpqa_diamond.csv',gpqa_choice_shuffle='private random.Random seeded from SHA256(42:RecordID); not fixed correct-option A',gpqa_answers=answer_counts,lcb_version='release_v6 medium',lcb_all383_test_md5_match=True,lcb_total_tests=sum(test_counts)),indent=2)+'\n')
