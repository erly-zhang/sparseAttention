import unittest
import torch
from attention import full_attention


def reference(q,k,v,pos,shared):
    b,h,n,d=q.shape
    group=h//k.shape[1]
    scores=q@k.repeat_interleave(group,1).transpose(-1,-2)/d**0.5
    causal=pos[:,:,None]>=pos[:,None,:]
    maps=scores.masked_fill(~causal[:,None],float('-inf')).softmax(-1)
    if shared:
        stitched=torch.stack([torch.stack([maps[bi,int(pos[bi,t])%h,t] for t in range(n)]) for bi in range(b)])
        maps=stitched[:,None].expand(-1,h,-1,-1)
    else:
        stitched=None
    output=maps@v.repeat_interleave(group,1)
    return output.transpose(1,2).reshape(b,n,h*d),stitched


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)

    def test_literal_reference_mha_gqa_and_short_sequences(self):
        for n in (1,3,9):
            for hk in (1,2,4):
                q=torch.randn(2,4,n,3,dtype=torch.float64)
                k,v=torch.randn(2,hk,n,3,dtype=torch.float64),torch.randn(2,hk,n,3,dtype=torch.float64)
                pos=torch.arange(n)[None].expand(2,-1)
                for mode in ('shared_map','dense_full'):
                    out,a,s=full_attention(q,k,v,pos,mode)
                    ref,ar=reference(q,k,v,pos,mode=='shared_map')
                    torch.testing.assert_close(out,ref)
                    if a is not None:
                        torch.testing.assert_close(a,ar)
                        torch.testing.assert_close(a.sum(-1),torch.ones(2,n,dtype=a.dtype))
                        self.assertEqual(torch.count_nonzero(a.triu(1)),0)
                    self.assertEqual(s['full_maps_computed'],4)
                    self.assertEqual(s['score_elements'],2*4*n*n)

    def test_original_position_ownership(self):
        q,k,v=[torch.randn(1,4,5,2,dtype=torch.float64) for _ in range(3)]
        pos=torch.tensor([[1,4,6,9,13]])
        out,a,_=full_attention(q,k,v,pos)
        expected,ea=reference(q,k,v,pos,True)
        torch.testing.assert_close(out,expected)
        torch.testing.assert_close(a,ea)

    def test_balanced_assignment_remainder(self):
        q,k,v=[torch.randn(1,4,11,2) for _ in range(3)]
        _,_,stats=full_attention(q,k,v,torch.arange(11)[None])
        self.assertEqual(stats['rows_per_head'],[[3],[3],[3],[2]])

    def test_no_future_leakage(self):
        q,k,v=[torch.randn(1,4,9,3,dtype=torch.float64) for _ in range(3)]
        pos=torch.arange(9)[None]
        out,a,_=full_attention(q,k,v,pos)
        q[:,:,5:]+=100; k[:,:,5:]-=300; v[:,:,5:]+=900
        changed,ca,_=full_attention(q,k,v,pos)
        torch.testing.assert_close(out[:,:5],changed[:,:5],rtol=0,atol=0)
        torch.testing.assert_close(a[:,:5],ca[:,:5],rtol=0,atol=0)

    def test_shared_map_is_not_repeated_output(self):
        q,k,v=[torch.randn(1,4,7,3,dtype=torch.float64) for _ in range(3)]
        pos=torch.arange(7)[None]
        out,_,_=full_attention(q,k,v,pos)
        dense,_=reference(q,k,v,pos,False)
        selected=dense.reshape(1,7,4,3)[0,torch.arange(7),torch.arange(7)%4]
        self.assertFalse(torch.allclose(out,selected.repeat(1,4)[None]))

    def test_gqa_values_remain_group_specific(self):
        q=torch.randn(1,4,7,3)
        k,v=torch.randn(1,2,7,3),torch.randn(1,2,7,3)
        out,_,_=full_attention(q,k,v,torch.arange(7)[None])
        out=out.reshape(1,7,4,3)
        torch.testing.assert_close(out[:,:,0],out[:,:,1],rtol=0,atol=0)
        torch.testing.assert_close(out[:,:,2],out[:,:,3],rtol=0,atol=0)
        self.assertFalse(torch.allclose(out[:,:,0],out[:,:,2]))

    def test_single_head_is_dense(self):
        q,k,v=[torch.randn(1,1,7,3) for _ in range(3)]
        pos=torch.arange(7)[None]
        a,_,_=full_attention(q,k,v,pos)
        b,_,_=full_attention(q,k,v,pos,'dense_full')
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_bf16_output_finite(self):
        q,k,v=[torch.randn(1,4,9,8,dtype=torch.bfloat16) for _ in range(3)]
        out,a,_=full_attention(q,k,v,torch.arange(9)[None])
        self.assertEqual(out.dtype,torch.bfloat16)
        self.assertEqual(a.dtype,torch.float32)
        self.assertTrue(torch.isfinite(out).all())

    def test_rejects_invalid_inputs(self):
        q,k,v=[torch.randn(1,4,5,2) for _ in range(3)]
        for pos in (torch.tensor([[0,1,1,3,4]]),torch.arange(5)[None].float()):
            with self.assertRaises(ValueError):
                full_attention(q,k,v,pos)
        with self.assertRaises(ValueError):
            full_attention(q,k[:,:3],v[:,:3],torch.arange(5)[None])
        q[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            full_attention(q,k,v,torch.arange(5)[None])

    def test_tiny_llama_rope_cache_dense_parity_and_decode(self):
        from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
        from adapter import install
        cfg=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval()
        original=model.model.layers[0].self_attn.forward
        ids=torch.arange(7)[None]; pos=torch.arange(7)[None]
        with torch.inference_mode():
            baseline=model(input_ids=ids,position_ids=pos,past_key_values=DynamicCache(),use_cache=True)
            for mode in ('dense_full','shared_map'):
                state=install(model,mode,layers=(0,2))
                cache=DynamicCache()
                out=model(input_ids=ids,position_ids=pos,past_key_values=cache,use_cache=True)
                self.assertTrue(torch.isfinite(out.logits).all())
                if mode=='dense_full':
                    torch.testing.assert_close(out.logits,baseline.logits,rtol=1e-5,atol=1e-6)
                self.assertEqual(set(state['prefill']),{0,2})
                saved=dict(state['prefill'])
                layer=model.model.layers[0].self_attn
                x=torch.randn(1,1,32)
                copies=[DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in cache.to_legacy_cache())) for _ in range(2)]
                a=layer(x,position_ids=torch.tensor([[7]]),past_key_value=copies[0],use_cache=True)
                b=original(x,position_ids=torch.tensor([[7]]),past_key_value=copies[1],use_cache=True)
                torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
                for j in range(3):
                    decoded=model(input_ids=torch.tensor([[1]]),position_ids=torch.tensor([[7+j]]),past_key_values=cache,use_cache=True)
                    self.assertTrue(torch.isfinite(decoded.logits).all())
                self.assertEqual(state['prefill'],saved)
                self.assertTrue(all(t.shape==(1,2,10,8) for t in cache.key_cache+cache.value_cache))
                state['close']()
                self.assertEqual(model.model.layers[0].self_attn.forward,original)

    def test_adapter_rejects_padding_and_duplicate_install(self):
        from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
        from adapter import install
        cfg=LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2)
        model=LlamaForCausalLM(cfg).eval()
        state=install(model,layers=(0,))
        with self.assertRaises(ValueError):
            install(model,layers=(0,))
        with torch.inference_mode(),self.assertRaises(ValueError):
            model.model.layers[0].self_attn(torch.randn(1,3,16),position_ids=torch.arange(3)[None],
                attention_mask=torch.tensor([[1,1,0]]),past_key_value=DynamicCache(),use_cache=True)
        state['close']()


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
