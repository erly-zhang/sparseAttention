import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prefill_token_head_routing_20260924/layer_output_comparison'))
from metrics import compare_outputs

class MetricsTests(unittest.TestCase):
    def test_identity_padding_and_chunking(self):
        a=torch.tensor([[[1.,2.,3.],[0.,0.,0.],[float('nan'),1.,2.]]]);mask=torch.tensor([[True,True,False]])
        r=compare_outputs(a,a,mask,chunk_tokens=1)
        self.assertEqual(r['valid_tokens'],2);self.assertEqual(r['relative_l2'],0);self.assertAlmostEqual(r['kl_dense_to_method'],0)
        self.assertAlmostEqual(r['mean_token_cosine'],1,places=6)  # float32 cosine roundoff
    def test_direction_and_l2(self):
        a=torch.tensor([[[0.,1.]]]);b=torch.tensor([[[2.,0.]]]);mask=torch.ones(1,1,dtype=torch.bool)
        r=compare_outputs(a,b,mask)
        p=a.double().softmax(-1);q=b.double().softmax(-1)
        self.assertAlmostEqual(r['kl_dense_to_method'],float((p*(p.log()-q.log())).sum()),places=6)
        self.assertAlmostEqual(r['relative_l2'],5**.5)
    def test_zero_reference(self):
        a=torch.zeros(1,1,2);b=torch.ones_like(a);mask=torch.ones(1,1,dtype=torch.bool)
        r=compare_outputs(a,b,mask);self.assertIsNone(r['relative_l2']);self.assertEqual(r['mean_token_cosine'],0)
if __name__=='__main__':unittest.main(verbosity=2)
