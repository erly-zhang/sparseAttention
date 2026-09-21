"""CPU-only orchestration contract tests using temporary synthetic records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.random_score_residual_20260910.schedule import (
    METHOD, preflight, save, task_config, validate)


class Contracts(unittest.TestCase):
    def test_missing_artifacts_fail_closed(self):
        result = preflight({})
        self.assertFalse(result['ready'])
        self.assertIn('reference_metrics', result['missing'])
        self.assertIn('group_config', result['missing'])

    def test_immutable_records(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'gate.json'
            save(p, {'passed': True})
            save(p, {'passed': True})
            with self.assertRaises(RuntimeError):save(p, {'passed': False})

    def test_yaml_preserves_prompt_metric_and_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'source'
            source.mkdir()
            (source/'kv_retrieval.yaml').write_text('task: kv_retrieval\ndataset_path: /old/loader.py\ndataset_kwargs:\n  data_dir: /old/data\ndoc_to_text: "{{context}} {{input}}"\nmetric: !function metrics.score\ngeneration_kwargs:\n  max_new_tokens: 128\n')
            (source/'metrics.py').write_text('def score(x): return x\n')
            c={'task_config_dir':str(source),'dataset_loader':'/restored/loader.py'}
            result=task_config(c,root/'new',[{'context':'fixture','input':'key'}])
            text=(result/'kv_retrieval.yaml').read_text()
            self.assertIn('!function',text)
            self.assertIn('metrics.score',text)
            self.assertIn('{{context}} {{input}}',text)
            self.assertIn('max_new_tokens: 128',text)
            self.assertIn('/restored/loader.py',text)
            self.assertNotIn('/old/',text)
            self.assertEqual((source/'metrics.py').read_bytes(),(result/'metrics.py').read_bytes())

    def test_input_pair_and_smoke_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ref=[{'input_ids_sha256':f'{i:064x}','input_tokens':100+i,
                  'selected_token_pairs':500+i,'causal_token_pairs':1000+i} for i in range(497)]
            rp=root/'reference.jsonl'
            rp.write_text(''.join(json.dumps(r)+'\n' for r in ref))
            c={k:str(rp) for k in ('reference_metrics','old_random_metrics','shared_metrics')}
            out=root/'output'
            out.mkdir()
            summary={'method':METHOD,'input_alignment':{'status':'passed','count':3,'reference_count':497},
                'run_args':{'max_new_tokens':128,'max_length':131072,'seed':42,'chat':True},
                'method_metadata':{'profile_only':False},
                'official_lm_eval_results':{'kv_retrieval':{'get_score_one_kv_retrieval,none':1/3}}}
            (out/'summary.json').write_text(json.dumps(summary))
            (out/'lm_eval_results.json').write_text(json.dumps({'samples':{'kv_retrieval':[
                {'doc_hash':str(i),'get_score_one_kv_retrieval':int(i==0)} for i in range(3)]}}))
            rr=[dict(r,profile_only=False,prefill_latency_sec=1.,decode_latency_sec=.1,total_latency_sec=1.1,
                     peak_memory_bytes=100,smoke_indices_and_finite_logits_passed=True) for r in ref[:3]]
            def write(records):
                (out/'online_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            write(rr[::-1])
            validate(out,c,0,3,'smoke')  # canonical identity sets tolerate batch ordering
            bad=copy.deepcopy(rr)
            bad[1]=copy.deepcopy(bad[0])
            write(bad)
            with self.assertRaises(AssertionError):validate(out,c,0,3,'smoke')
            bad=copy.deepcopy(rr)
            bad[1]['selected_token_pairs']+=1
            write(bad)
            with self.assertRaises(AssertionError):validate(out,c,0,3,'smoke')
            bad=copy.deepcopy(rr)
            bad[1]['smoke_indices_and_finite_logits_passed']=False
            write(bad)
            with self.assertRaises(AssertionError):validate(out,c,0,3,'smoke')


if __name__ == '__main__':unittest.main()
