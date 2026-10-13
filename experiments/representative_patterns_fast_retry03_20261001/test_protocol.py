"""CPU tests for scenario/head provenance and the approval-before-work boundary."""
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch
from protocol import configuration
from run import fixed_random_heads
from baselines import baseline_attention
from planning import work_plan
from collections import Counter


class ProtocolTests(unittest.TestCase):
    def test_gpu_proof_path_preserves_path_type_and_job_name(self):
        from run import gpu_proof_path
        path=gpu_proof_path('12345')
        self.assertIsInstance(path,Path)
        self.assertEqual(path.name,'gpu-parity-12345.json')
        self.assertEqual(path.parent.name,'outputs')

    def test_matrix_and_all_layers(self):
        c=configuration()
        self.assertEqual(len(c['methods']),7)
        self.assertEqual(c['layers'],list(range(32)))
        self.assertEqual(c['baseline_layers'],c['layers'])
        for strategy in ('coverage','random'):
            for mask in ('token_vertical','token_vertical_slash'):
                self.assertEqual(c['scenarios'][strategy+'_'+mask],
                    {'head_strategy':strategy,'selector_method':'representative_'+mask})
        self.assertEqual(c['formal']['samples_per_task'],50)

    def test_plan_covers_all_pairs_and_balances_each_task(self):
        c=configuration()
        m=json.loads((Path(__file__).parent/'inputs/manifest.json').read_text())
        plan=work_plan(c,m)
        self.assertEqual(len(plan),70)
        pairs=[(key,row['method']) for row in plan for key in row['keys']]
        self.assertEqual(len(pairs),1400)
        self.assertEqual(len(set(pairs)),1400)
        self.assertEqual(set(pairs),{(r['key'],method) for r in m['evaluation'] for method in c['methods']})
        for i,row in enumerate(plan):
            self.assertEqual(row['worker'],i)
            self.assertEqual(len(row['keys']),20)
            self.assertEqual(set(row['task_counts'].values()),{5})
        self.assertIsNone(c['resource_policy']['aggregate_gpu_hours_limit'])

    def test_random_heads_fixed_across_masks_and_evaluation_examples(self):
        c=configuration()
        a=fixed_random_heads(c);b=fixed_random_heads(c)
        self.assertEqual(a,b);self.assertEqual(set(a),set(map(str,range(32))))
        self.assertTrue(all(type(h) is int and 0<=h<32 for h in a.values()))
        changed=dict(c,calibration=dict(c['calibration'],random_head_seed=c['calibration']['random_head_seed']+1))
        self.assertNotEqual(a,fixed_random_heads(changed))

    def test_submit_checks_permission_before_preflight_and_scheduler(self):
        import submit
        with patch('submit.require_approval',side_effect=PermissionError('pending')),\
             patch('submit.preflight') as preflight,patch('submit.subprocess.run') as scheduler:
            with self.assertRaises(PermissionError):submit.submit()
            preflight.assert_not_called();scheduler.assert_not_called()

    def test_baseline_minference_uses_own_gqa_kv_head(self):
        calls=[]
        def official(q,k,v,head,layer,config):
            calls.append((head,layer,float(k[0,0,0,0])))
            return v.expand_as(q)
        fake=types.ModuleType('minference.modules.minference_forward')
        fake.minference_prefill_kernel=official
        q=torch.zeros(1,9,4,8);k=torch.zeros(1,9,2,8);k[:,:,1]=7
        v=torch.ones_like(k);v[:,:,1]=13
        options={'minference':{'pattern_file':'not-read','ratio':1}}
        with patch.dict('sys.modules',{'minference.modules.minference_forward':fake}),\
             patch('baselines.pattern',return_value=[{},{}]):
            out,stats=baseline_attention(q,k,v,'minference',1,options)
        self.assertEqual(calls,[(0,1,0),(1,1,0),(2,1,7),(3,1,7)])
        torch.testing.assert_close(out[:,:,:2],torch.ones_like(out[:,:,:2]))
        torch.testing.assert_close(out[:,:,2:],torch.full_like(out[:,:,2:],13))
        self.assertFalse(stats['probabilities_shared'])


if __name__=='__main__':
    unittest.main()
