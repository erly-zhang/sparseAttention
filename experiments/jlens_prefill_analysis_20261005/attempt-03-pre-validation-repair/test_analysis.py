"""CPU mathematical and integration checks; no real-model execution."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from jlens import fit,JacobianLens
from common import positions_for,Llama44LensModel,atomic_torch_save
from metrics import distribution_metrics,ranks_of,layer_metrics
from prepare import evidence_positions
from progress import PeriodicSaver


class AnalysisTests(unittest.TestCase):
    def test_dense_parity_checks_decision_and_distribution(self):
        from validation import dense_parity
        dense=torch.tensor([[2.,1.,0.]])
        result=dense_parity(dense,dense+torch.tensor([[.01,-.01,.01]]))
        self.assertTrue(result['first_token_equal'])
        self.assertLess(result['js_nats'],.001)
        # Small per-logit differences can flip a tied decision: reject them.
        with self.assertRaisesRegex(ValueError,'first token'):
            dense_parity(torch.tensor([[1.,1.01,0.]]),torch.tensor([[1.02,1.,0.]]))
        # Per-logit tolerance alone can accept a material distribution shift.
        with self.assertRaisesRegex(ValueError,'distribution'):
            dense_parity(torch.tensor([[0.,0.,0.]]),torch.tensor([[.19,-.19,0.]]))

    def test_distribution_identity_and_constant_shift(self):
        values=torch.tensor([[3.,1.,0.],[-2.,4.,7.]])
        metrics=distribution_metrics(values,values+7,top_k=2)
        torch.testing.assert_close(metrics['js_nats'],torch.zeros(2),atol=1e-7,rtol=0)
        torch.testing.assert_close(metrics['kl_dense_to_sparse_nats'],torch.zeros(2),atol=1e-7,rtol=0)
        self.assertEqual(metrics['top_k_overlap'].tolist(),[1.,1.])

    def test_js_bound_and_symmetry(self):
        left=torch.tensor([[100.,-100.,-100.]])
        right=torch.tensor([[-100.,100.,-100.]])
        a=distribution_metrics(left,right,top_k=2)['js_nats']
        b=distribution_metrics(right,left,top_k=2)['js_nats']
        torch.testing.assert_close(a,b)
        self.assertAlmostEqual(float(a),float(torch.tensor(2.).log()),places=6)

    def test_rank_definition_and_ties(self):
        ranks=ranks_of(torch.tensor([[5.,5.,2.,1.]]),torch.tensor([[0,1,2,3]]))
        self.assertEqual(sorted(ranks[0,:2].tolist()),[1,2])
        self.assertEqual(ranks[0,2:].tolist(),[3,4])

    def test_small_attention_mass_can_drop_distinct_value(self):
        # Exact fixed-QKV identity motivating value-sensitive diagnostics.
        weights=torch.tensor([.98,.02])
        values=torch.tensor([[0.,0.],[0.,100.]])
        dense=weights@values
        sparse=values[0]
        torch.testing.assert_close(dense-sparse,.02*(values[1]-values[0]))
        self.assertGreater(float((dense-sparse).norm()),1.)

    def test_sampling_preserves_original_positions(self):
        options=dict(prefix_positions=3,tail_positions=4,uniform_positions=5)
        sampled=positions_for(100,options,evidence=[30,31])
        self.assertEqual(sampled,sorted(set(sampled)))
        self.assertTrue({0,1,2,30,31,96,97,98,99}.issubset(sampled))
        self.assertTrue(all(0<=p<100 for p in positions_for(2,options)))

    def test_annotation_rejects_token_mismatch(self):
        tokenizer=lambda *args,**kwargs:dict(input_ids=[99],offset_mapping=[(0,3)])
        with self.assertRaisesRegex(ValueError,'different input IDs'):
            evidence_positions(tokenizer,'abc',[1],'abc',2)

    def test_checkpoint_hardlink_survives_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.pt'
            atomic_torch_save(p,dict(n=1))
            saver=PeriodicSaver(directory)
            saver.snapshot()
            preserved=next((Path(directory)/'periodic-saves').glob('checkpoint-*.pt'))
            atomic_torch_save(p,dict(n=2))
            self.assertEqual(torch.load(p,weights_only=True)['n'],2)
            self.assertEqual(torch.load(preserved,weights_only=True)['n'],1)

    def test_original_fitting_with_protocol_and_chunked_readout(self):
        import importlib.util
        from common import config
        p=Path(config()['jlens_repository'])/'tests/tiny.py'
        spec=importlib.util.spec_from_file_location('_jlens_tiny',p)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tiny=module.TinyDecoder(n_layers=4,d_model=8).requires_grad_(False)
        class TextInterface:
            layers=tiny.layers
            norm=tiny.norm
            def __call__(self,input_ids,**kwargs):
                return tiny.forward(input_ids)
        hf=SimpleNamespace(model=TextInterface(),lm_head=tiny.lm_head,device=torch.device('cpu'),
                           config=SimpleNamespace(num_hidden_layers=4,hidden_size=8))
        wrapper=Llama44LensModel(hf,tiny.tokenizer)
        lens=fit(wrapper,['abcdefghij '*5],dim_batch=4,max_seq_len=32)
        hidden=torch.randn(7,8)
        altered=hidden+torch.randn_like(hidden)*.2
        chunked=layer_metrics(wrapper,lens,1,hidden,altered,3,2)
        direct=distribution_metrics(wrapper.unembed(lens.transport(hidden,1)),
                                    wrapper.unembed(lens.transport(altered,1)),3)
        torch.testing.assert_close(torch.tensor(chunked['js_nats']),direct['js_nats'])
        same=layer_metrics(wrapper,lens,3,hidden,hidden,3,2)
        self.assertEqual(same['js_nats'],[0.]*7)


if __name__=='__main__': unittest.main()
