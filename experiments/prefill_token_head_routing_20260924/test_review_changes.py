"""Pending regression tests for static review changes. NOT RUN on 2026-09-25."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validate_results import input_hash, validate


def fixture(root):
    for mode in ('dense','q_only','q_kv'):
        cell = root/mode/'passkey'; cell.mkdir(parents=True)
        samples = [{'doc_id':0,'doc_hash':'a'*64,'prompt_hash':'b'*64,'target_hash':'c'*64}]
        (cell/'lm_eval_results.json').write_text(json.dumps({'samples':{'passkey':samples},'results':{'passkey':{'score':0.0}}}))
        row = {'mode':mode,'input_ids':[2,3], 'input_ids_sha256':input_hash([2,3]),
               'generated_token_ids':[4], 'generated_length':1,'stop_reason':'length_limit',
               'prefill_seconds':0.1,'decode_forward_seconds':0.0,'total_seconds':0.2}
        metric = {'input_ids_sha256':row['input_ids_sha256'],'input_tokens':2,'generated_tokens':1}
        (cell/'online_metrics.jsonl').write_text(json.dumps(metric)+'\n')
        (cell/'routing_generations.jsonl').write_text(json.dumps(row)+'\n')


class ResultIntegrityTests(unittest.TestCase):
    def test_valid_records_and_corruption_rejection(self):
        for corruption in (None,'doc_identity','saved_input','generation_length','missing','duplicate'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture(root)
                cell=root/'q_only/passkey'
                side=cell/'routing_generations.jsonl'
                if corruption == 'doc_identity':
                    p=cell/'lm_eval_results.json'; data=json.loads(p.read_text())
                    data['samples']['passkey'][0]['doc_hash']='d'*64; p.write_text(json.dumps(data))
                elif corruption in ('saved_input','generation_length'):
                    row=json.loads(side.read_text())
                    if corruption == 'saved_input': row['input_ids']=[3,2]
                    else: row['generated_length']=2
                    side.write_text(json.dumps(row)+'\n')
                elif corruption == 'missing': side.write_text('')
                elif corruption == 'duplicate': side.write_text(side.read_text()*2)
                with patch('validate_results.constant', return_value={'passkey':1}):
                    if corruption is None: self.assertEqual(len(validate(root)),3)
                    else:
                        with self.assertRaises(ValueError): validate(root)


class BridgeAndGuardTests(unittest.TestCase):
    def test_position_rejection_before_cache_update(self):
        import torch
        from transformers import DynamicCache
        from test_routing import small_model, DEVICE
        from routing import RoutingController
        for mode in ('dense','q_only','q_kv'):
            model=small_model(); ctl=RoutingController(model,mode)
            ids=torch.tensor([[2,3]],device=DEVICE); mask=torch.ones_like(ids)
            positions=torch.tensor([[1,0]],device=DEVICE)
            cache=DynamicCache(config=model.config)
            try:
                with self.assertRaisesRegex(ValueError,'strictly increasing'), torch.no_grad(), ctl.phase('prefill',mask.bool(),positions):
                    model(ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True)
                self.assertEqual(cache.get_seq_length(),0)
            finally: ctl.close()

    def test_streamed_input_retention(self):
        import torch
        from test_routing import small_model, DEVICE
        from generation_bridge import GenerationBridge
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            model=small_model(); bridge=GenerationBridge(model,'q_only',path,retain_input_ids=False)
            ids=torch.tensor([[2,3,4]],device=DEVICE)
            try:
                model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=2,pad_token_id=0)
                self.assertNotIn('input_ids',bridge.rows[0])
                saved=json.loads(path.read_text())
                self.assertEqual(saved['input_ids'],[2,3,4])
                self.assertEqual(saved['input_ids_sha256'],input_hash([2,3,4]))
                self.assertIn('generation_config',saved)
            finally: bridge.close()


class RecorderTests(unittest.TestCase):
    def test_bridge_clock_and_single_token_counts(self):
        from types import SimpleNamespace
        from recorder import make_recorder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = SimpleNamespace(rows=[],output=root/'sidecar.jsonl')
            detail = dict(valid_prompt_tokens=1,input_tokens=1,input_ids_sha256='a'*64,
                          generated_length=2,prefill_seconds=0.1,decode_forward_seconds=0.2,
                          total_seconds=0.4,peak_allocated_bytes=1024,stop_reason='length_limit',
                          prefill_pairs_by_layer=[{'eligible_pairs':1},{'eligible_pairs':1}])
            def generation(*args,**kwargs):
                bridge.rows.append(copy.deepcopy(detail))
                return 'unchanged output'
            model = SimpleNamespace(generate=generation,config=SimpleNamespace(num_hidden_layers=2,num_attention_heads=4))
            class Base:
                def __init__(self):
                    self.model=model; self._original_generate=model.generate
                    self.rows=[]; self._installed=False; self.method='q_only'
                    self.context={'benchmark':'infinitebench','task':'passkey'}
                    self.output_path=root/'metrics.jsonl'
                def summary(self):
                    return {'count':len(self.rows)}
            rec=make_recorder(Base,lambda:bridge)()
            rec.install()
            self.assertEqual(model.generate(),'unchanged output')
            row=json.loads(rec.output_path.read_text())
            self.assertEqual(row['prefill_latency_sec'],0.1)
            self.assertEqual(row['total_latency_sec'],0.4)
            self.assertEqual(row['decode_forward_latency_sec'],0.2)
            self.assertEqual(row['selected_token_pairs'],2)
            self.assertEqual(row['causal_token_pairs'],8)
            self.assertEqual(row['global_token_sparsity'],0.75)
            self.assertEqual(rec.summary()['avg_decode_forward_latency_sec'],0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
