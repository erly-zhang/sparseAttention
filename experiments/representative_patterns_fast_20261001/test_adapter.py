"""Tiny randomly initialized CPU model validates cache and adapter lifecycle only."""
import unittest
from unittest.mock import patch
import torch
from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
from adapter import install


class AdapterTests(unittest.TestCase):
    def model(self):
        c=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        c._attn_implementation='eager'
        return LlamaForCausalLM(c).eval()

    def test_prefill_dispatch_dense_decode_and_restore(self):
        model=self.model()
        original=model.model.layers[0].self_attn.forward
        def fake(q,k,v,positions,mode,**kwargs):
            self.assertEqual(mode,'representative_token')
            self.assertEqual(kwargs['head'],3)
            self.assertEqual(kwargs['layer'],0)
            self.assertEqual(tuple(k.shape),(1,2,7,8))
            return q.transpose(1,2).flatten(-2),None,dict(mode=mode)
        state=install(model,'representative_token',layers=(0,),heads={'0':3},options={})
        cache=DynamicCache()
        try:
            with patch('adapter.full_attention',side_effect=fake) as dispatch,torch.no_grad():
                model.model(input_ids=torch.arange(7)[None],position_ids=torch.arange(7)[None],
                    attention_mask=torch.ones(1,7,dtype=torch.long),past_key_values=cache,use_cache=True)
                model.model(input_ids=torch.tensor([[8]]),position_ids=torch.tensor([[7]]),
                    attention_mask=torch.ones(1,8,dtype=torch.long),past_key_values=cache,use_cache=True)
            self.assertEqual(dispatch.call_count,1)
            self.assertEqual(state['decode_calls'],{0:1})
            self.assertTrue(all(k.shape==v.shape==(1,2,8,8) for k,v in zip(cache.key_cache,cache.value_cache)))
        finally:
            state['close']()
        self.assertEqual(model.model.layers[0].self_attn.forward,original)
        self.assertFalse(model._shared_map_installed)

    def test_repeated_install_rejected(self):
        model=self.model()
        state=install(model,'dense_full',layers=(0,))
        try:
            with self.assertRaises(ValueError): install(model,'dense_full',layers=(0,))
        finally:
            state['close']()

    def test_sparse_requires_fixed_heads(self):
        with self.assertRaises(ValueError): install(self.model(),'representative_token',layers=(0,))


if __name__=='__main__':
    unittest.main()
