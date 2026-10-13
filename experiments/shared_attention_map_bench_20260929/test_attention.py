import unittest
import torch
from attention import full_attention
from reference_attention import full_attention as reference


class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def tensors(self,h=4,hk=2,n=13,dtype=torch.float64):
        return (torch.randn(2,h,n,8,dtype=dtype),torch.randn(2,hk,n,8,dtype=dtype),
                torch.randn(2,hk,n,8,dtype=dtype),torch.arange(n)[None].repeat(2,1))

    def test_blocked_equals_literal(self):
        for h,hk in ((4,4),(4,2),(4,1),(1,1)):
            for n in (1,7,17):
                for mode in ('dense_full','shared_map'):
                    args=self.tensors(h,hk,n)
                    expected,matrix,stats=reference(*args,mode)
                    for block in (1,3,8,32):
                        actual,stitched,got=full_attention(*args,mode,block,True)
                        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                        if matrix is not None:
                            torch.testing.assert_close(stitched,matrix,rtol=1e-12,atol=1e-12)
                        for key in ('score_elements','rows_per_head','full_causal_pairs','stitched_causal_pairs'):
                            self.assertEqual(got[key],stats[key])

    def test_noncontiguous_original_positions(self):
        q,k,v,pos=self.tensors()
        pos=pos*3+7
        a,m,_=full_attention(q,k,v,pos,block_size=3,return_map=True)
        b,r,_=reference(q,k,v,pos)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(m,r)

    def test_future_suffix_cannot_change_past(self):
        q,k,v,pos=self.tensors()
        a,_,_=full_attention(q,k,v,pos,block_size=3)
        q[:,:,7:]+=10
        k[:,:,7:]-=20
        v[:,:,7:]+=30
        b,_,_=full_attention(q,k,v,pos,block_size=3)
        torch.testing.assert_close(a[:,:7],b[:,:7])

    def test_bf16_parity_and_no_full_map(self):
        args=self.tensors(dtype=torch.bfloat16)
        a,m,_=full_attention(*args,block_size=3)
        b,_,_=reference(*args)
        self.assertIsNone(m)
        self.assertEqual(a.dtype,torch.bfloat16)
        torch.testing.assert_close(a,b,atol=0.02,rtol=0.01)

    def test_validation_rejects_bad_inputs(self):
        q,k,v,pos=self.tensors()
        with self.assertRaises(ValueError): full_attention(q,k,v,pos,block_size=0)
        with self.assertRaises(ValueError): full_attention(q,k,v,pos.float())
        with self.assertRaises(ValueError): full_attention(q,k,v,pos.flip(-1))
        q[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError): full_attention(q,k,v,pos)

    def test_gqa_output_groups(self):
        q,k,v,pos=self.tensors()
        out,_,_=full_attention(q,k,v,pos,block_size=3)
        out=out.view(2,13,4,8)
        self.assertTrue(torch.equal(out[:,:,0],out[:,:,1]))
        self.assertFalse(torch.equal(out[:,:,0],out[:,:,2]))

    def test_llama_cache_decode_and_dense_control(self):
        from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
        from adapter import install
        cfg=LlamaConfig(vocab_size=101,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval()
        ids=torch.tensor([[1,3,5,7,9,11,13]])
        with torch.no_grad():
            baseline=model(ids,use_cache=True).logits
            for method in ('dense_full','shared_map'):
                state=install(model,method,layers=(0,2))
                cache=DynamicCache()
                try:
                    out=model(ids,past_key_values=cache,use_cache=True)
                    if method=='dense_full':
                        torch.testing.assert_close(out.logits,baseline,atol=1e-6,rtol=1e-5)
                    first=out.logits[:,-1].argmax(-1,keepdim=True)
                    model(first,past_key_values=cache,use_cache=True)
                    self.assertEqual(cache.get_seq_length(),8)
                    self.assertEqual(state['decode_calls'],{0:1,2:1})
                    self.assertEqual(set(state['prefill']),{0,2})
                finally:
                    state['close']()


if __name__=='__main__': unittest.main()
