import unittest
import torch
from attention import full_attention,nucleus_mask
from reference_attention import full_attention as reference


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def tensors(self,h=4,hk=2,n=13,dtype=torch.float64):
        return (torch.randn(2,h,n,8,dtype=dtype),torch.randn(2,hk,n,8,dtype=dtype),
            torch.randn(2,hk,n,8,dtype=dtype),torch.arange(n)[None].repeat(2,1))

    def test_blocked_literal(self):
        for h,hk in ((4,4),(4,2),(4,1),(1,1)):
            for n in (1,7,17):
                for mode in ('dense_full','shared_topp'):
                    args=self.tensors(h,hk,n)
                    expected,mask,_=reference(*args,mode)
                    for block in (1,3,8,32):
                        actual,m,stat=full_attention(*args,mode,block,True)
                        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                        if mask is not None:
                            self.assertTrue(torch.equal(m,mask))
                            self.assertEqual(stat['mask_selected_pairs'],int(m.sum()))

    def test_top_p_one_is_dense(self):
        for dtype in (torch.float64,torch.float32,torch.bfloat16):
            args=self.tensors(dtype=dtype)
            a,_,_=full_attention(*args,mode='dense_full',block_size=3)
            b,m,_=full_attention(*args,block_size=3,return_map=True,top_p=1)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(torch.equal(m,torch.ones_like(m).tril()))

    def test_noncontiguous_original_positions(self):
        q,k,v,p=self.tensors()
        p=p*3+7
        a,m,s=full_attention(q,k,v,p,block_size=3,return_map=True)
        b,r,_=reference(q,k,v,p)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.equal(m,r))
        self.assertEqual(s['rows_per_head'],[(p.remainder(4)==h).sum(-1).tolist() for h in range(4)])

    def test_causal_suffix_invariance(self):
        q,k,v,p=self.tensors()
        a,m,_=full_attention(q,k,v,p,block_size=3,return_map=True)
        q[:,:,7:]+=10
        k[:,:,7:]-=20
        v[:,:,7:]+=30
        b,r,_=full_attention(q,k,v,p,block_size=3,return_map=True)
        torch.testing.assert_close(a[:,:7],b[:,:7])
        self.assertTrue(torch.equal(m[:,:7],r[:,:7]))

    def test_bf16_parity(self):
        args=self.tensors(dtype=torch.bfloat16)
        a,m,_=full_attention(*args,block_size=3)
        b,_,_=reference(*args)
        self.assertIsNone(m)
        self.assertEqual(a.dtype,torch.bfloat16)
        torch.testing.assert_close(a,b,atol=0.02,rtol=0.01)

    def test_heads_keep_own_queries_despite_same_kv(self):
        args=self.tensors(hk=1)
        out,_,_=full_attention(*args,block_size=3)
        out=out.view(2,13,4,8)
        self.assertFalse(torch.equal(out[:,:,0],out[:,:,1]))

    def test_minimal_prefix_includes_crossing_token(self):
        p=torch.tensor([[0.5,0.3,0.15,0.05]],dtype=torch.float64)
        m,mass,prev=nucleus_mask(p,torch.ones_like(p,dtype=torch.bool),0.9)
        self.assertEqual(m.tolist(),[[True,True,True,False]])
        self.assertAlmostEqual(float(mass),0.95)
        self.assertAlmostEqual(float(prev),0.8)

    def test_threshold_equality_excludes_next_token(self):
        p=torch.tensor([[0.5,0.25,0.25]],dtype=torch.float64)
        m,_,_=nucleus_mask(p,torch.ones_like(p,dtype=torch.bool),0.75)
        self.assertEqual(m.tolist(),[[True,True,False]])

    def test_stable_ties_key_order_and_causal_zeros(self):
        p=torch.tensor([[0.25]*4+[0.0]],dtype=torch.float64)
        allowed=torch.tensor([[True]*4+[False]])
        m,_,_=nucleus_mask(p,allowed,0.5)
        self.assertEqual(m.tolist(),[[True,True,False,False,False]])

    def test_per_row_selection_not_global(self):
        p=torch.tensor([[0.99,0.01,0],[0.5,0.3,0.2]],dtype=torch.float64)
        m,_,_=nucleus_mask(p,torch.ones_like(p,dtype=torch.bool),0.95)
        self.assertEqual(m.sum(-1).tolist(),[1,3])

    def test_reject_invalid(self):
        q,k,v,p=self.tensors()
        for prob in (0,-1,1.1,float('nan')):
            with self.assertRaises(ValueError): full_attention(q,k,v,p,top_p=prob)
        with self.assertRaises(ValueError): full_attention(q,k,v,p,block_size=0)
        with self.assertRaises(ValueError): full_attention(q,k,v,p.float())
        with self.assertRaises(ValueError): full_attention(q,k,v,p.flip(-1))
        q[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError): full_attention(q,k,v,p)

    def test_no_empty_row_and_illegal_probability(self):
        with self.assertRaises(ValueError): nucleus_mask(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool),0.95)
        with self.assertRaises(ValueError): nucleus_mask(torch.ones(1,2)/2,torch.tensor([[True,False]]),0.95)

    def test_llama_cache_decode_dense_control(self):
        from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
        from adapter import install
        cfg=LlamaConfig(vocab_size=101,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval()
        ids=torch.tensor([[1,3,5,7,9,11,13]])
        with torch.no_grad():
            baseline=model(ids,use_cache=True).logits
            for method in ('dense_full','shared_topp'):
                state=install(model,method,layers=(0,2))
                cache=DynamicCache()
                try:
                    out=model(ids,past_key_values=cache,use_cache=True)
                    if method=='dense_full':
                        torch.testing.assert_close(out.logits,baseline,atol=1e-6,rtol=1e-5)
                    model(out.logits[:,-1].argmax(-1,keepdim=True),past_key_values=cache,use_cache=True)
                    self.assertEqual(cache.get_seq_length(),8)
                    self.assertEqual(state['decode_calls'],{0:1,2:1})
                    self.assertEqual(set(state['prefill']),{0,2})
                finally:
                    state['close']()


if __name__=='__main__': unittest.main()
