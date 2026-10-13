import copy
import unittest
from protocol import TASKS,METHODS,LAYERS,MAX_NEW,assign_shards,validate_inputs,validate_pair,digest,render


def fixtures():
    records=[]
    for task in TASKS:
        for i in range(50):
            records.append(dict(key=task+'-'+str(i),task=task,source_index=i,dataset_id=i,
                input_tokens=99+i,before_truncation=99+i,input_ids_sha256=digest([task,i]),
                position_ids_sha256='pos',attention_mask_sha256='mask',rendered_prompt_sha256='prompt',
                raw_row_sha256='raw',answers=['yes'],max_new_tokens=MAX_NEW[task]))
    assignments,costs=assign_shards(records)
    for r in records: r['shard']=assignments[r['key']]
    manifest=dict(records=records,methods=list(METHODS),layers=list(LAYERS),
        truncation='none; complete examples only',model_context_limit=131072,shard_costs=costs)
    item=records[0]
    n=item['input_tokens']
    rows=[]
    for method in METHODS:
        r=dict(item,method=method,manifest_sha256='manifest',config_sha256='config',
            finite_all_steps=True,full_kv_cache=True,dense_decode=True,generated_ids=[2],
            eos_ids=[2],stop_reason='eos',decode_calls={str(i):0 for i in LAYERS},
            node='node',gpu='H100',torch='2',transformers='4',generation_seed=42)
        r['prefill']={str(i):dict(mode=method,full_maps_computed=32,score_elements=32*n*n,
            full_causal_pairs=32*n*(n+1)//2,all_keys_per_block=n,max_row_sum_error=0,
            rows_per_head=[[n//32+int(h<n%32)] for h in range(32)]) for i in LAYERS}
        rows.append(r)
    return manifest,item,rows


class ProtocolTests(unittest.TestCase):
    def test_complete_manifest_and_pair(self):
        manifest,item,rows=fixtures()
        validate_inputs(manifest)
        validate_pair(rows,item,'manifest','config')

    def test_stable_balanced_pair_assignment(self):
        manifest,_,_=fixtures()
        a,_=assign_shards(manifest['records'])
        b,_=assign_shards(list(reversed(manifest['records'])))
        self.assertEqual(a,b)
        self.assertEqual([list(a.values()).count(i) for i in range(4)],[50]*4)

    def test_mismatched_inputs_config_and_stop_rejected(self):
        _,item,rows=fixtures()
        for field in ('input_ids_sha256','position_ids_sha256','attention_mask_sha256',
                      'rendered_prompt_sha256','raw_row_sha256','manifest_sha256','config_sha256','gpu'):
            changed=copy.deepcopy(rows)
            changed[1][field]='DIFFERENT'
            with self.assertRaises(AssertionError,msg=field): validate_pair(changed,item,'manifest','config')
        rows[0]['generated_ids']=[3]
        with self.assertRaises(AssertionError): validate_pair(rows,item,'manifest','config')

    def test_duplicate_missing_and_truncated_rejected(self):
        manifest,item,rows=fixtures()
        with self.assertRaises(AssertionError): validate_pair([rows[0],rows[0]],item,'manifest','config')
        manifest['records'][0]['before_truncation']+=1
        with self.assertRaises(AssertionError): validate_inputs(manifest)

    def test_render_keeps_question_options_and_context(self):
        template={t:'{context}|{input}|{question}|{OPTION_A}|{OPTION_D}' for t in TASKS}
        row=dict(context='unchanged',input='question',options=['one','two','three','four'])
        self.assertEqual(render(row,'longbook_choice_eng',template),'unchanged|question|question|one|four')


if __name__=='__main__': unittest.main()
