import unittest,torch
from pursuit import pursuit,dense_span_energy

class Tests(unittest.TestCase):
    def test_orthogonal_nonnegative_and_negative_remainder(self):
        a=torch.eye(5);x=torch.tensor([[3.,2.,1.,-4.,0.],[0.,0.,0.,0.,0.]])
        r=pursuit(x,a,(1,3))
        self.assertTrue(torch.allclose(r[3]['reconstruction'][0],torch.tensor([3.,2.,1.,0.,0.])))
        self.assertEqual(int(r[3]['active_count'][1]),0)
        self.assertEqual(float(r[3]['energy_reconstructed'][1]),0.)
        self.assertTrue((r[3]['energy_reconstructed']>=r[1]['energy_reconstructed']).all())
    def test_correlated_atoms_joint_update_matches_scalar_reference(self):
        torch.manual_seed(3);a=torch.nn.functional.normalize(torch.randn(17,6),dim=-1);x=torch.randn(4,6)
        bat=pursuit(x,a,(3,5))
        for i in range(4):
            residual=x[i].clone();ids=[];c=torch.empty(0)
            for k in range(5):
                corr=a@residual
                if ids:corr[ids]=-float('inf')
                idx=int(corr.argmax());ids.append(idx);selected=a[ids];c=torch.cat([c,torch.zeros(1)])
                g=selected@residual;g[(c==0)&(g<0)]=0
                proj=g@selected;eta=(proj@residual)/(proj@proj).clamp_min(1e-30)
                for _ in range(20):
                    trial=(c+eta*g).clamp_min(0);new=x[i]-trial@selected
                    if new.square().sum()<=residual.square().sum()+1e-6*residual.square().sum():break
                    eta*=.5
                c=trial;residual=new
            self.assertTrue(torch.allclose(bat[5]['reconstruction'][i],x[i]-residual,atol=2e-6))
            self.assertTrue((bat[5]['coefficients']>=0).all())
    def test_dense_signed_span_is_orthogonal_split(self):
        a=torch.eye(3);delta=torch.tensor([[1.,2.,2.]])
        f=dense_span_energy(delta,a,torch.tensor([[0]]),torch.ones(1,1))
        self.assertAlmostEqual(float(f),1/9,places=6)
        self.assertTrue(torch.isnan(dense_span_energy(torch.zeros(1,3),a,torch.tensor([[0]]),torch.ones(1,1))).all())
    def test_rotated_control_preserves_dictionary_gram(self):
        torch.manual_seed(4);a=torch.randn(7,4);q,_=torch.linalg.qr(torch.randn(4,4))
        self.assertTrue(torch.allclose((a@q)@(a@q).T,a@a.T,atol=2e-5))

if __name__=='__main__':unittest.main()
