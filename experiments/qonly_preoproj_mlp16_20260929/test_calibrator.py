import unittest
from types import SimpleNamespace
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
from calibrator import PreProjectionMLP, ProjectionHooks, CaptureComplete, loss


class CalibratorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_identity_and_count(self):
        a=PreProjectionMLP()
        self.assertEqual(sum(p.numel() for p in a.parameters())*16,16844800)
        for dtype in (torch.float32,torch.bfloat16):
            x=torch.randn(1,3,4096).to(dtype)
            self.assertTrue(torch.equal(a(x),x))

    def test_gradient_and_learning(self):
        a=PreProjectionMLP(16,4)
        projection=nn.Linear(16,16,bias=False).requires_grad_(False)
        before=projection.weight.detach().clone()
        x=torch.randn(64,16,requires_grad=True)
        y=(x.detach()+0.2)
        optimizer=torch.optim.Adam(a.parameters(),lr=.01)
        initial=loss(a,projection,x,y).item()
        for _ in range(15):
            optimizer.zero_grad()
            value=loss(a,projection,x,y)
            value.backward()
            optimizer.step()
        self.assertLess(loss(a,projection,x,y).item(),initial)
        self.assertIsNone(x.grad)
        self.assertIsNone(projection.weight.grad)
        self.assertTrue(torch.equal(before,projection.weight))
        with self.assertRaises(ValueError):
            loss(a,nn.Linear(16,16),x,y)

    def test_preprojection_and_explicit_decode_bypass(self):
        projection=nn.Linear(4,4,bias=False)
        projection.weight.data.copy_(torch.diag(torch.tensor([1.,2.,3.,4.])))
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection))]))
        hooks=ProjectionHooks(model,[0])
        adapter=PreProjectionMLP(4,2)
        adapter.up.bias.data.fill_(1)
        hooks.adapters={0:adapter}
        hooks.enabled=True
        x=torch.randn(1,1,4)
        baseline=projection(x)
        hooks.prefill=True
        self.assertTrue(torch.allclose(projection(x),baseline+torch.tensor([1.,2.,3.,4.])))
        self.assertEqual(hooks.calls,{0:1})
        hooks.prefill=False
        self.assertTrue(torch.equal(projection(x),baseline))
        self.assertEqual(hooks.calls,{0:1})
        hooks.close()

    def test_capture_before_correction_and_stop(self):
        projection=nn.Linear(8,8)
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection))]))
        hooks=ProjectionHooks(model,[0])
        hooks.prefill=True
        hooks.enabled=True
        hooks.adapters={0:PreProjectionMLP(8,2)}
        hooks.capture={0}
        hooks.indices=[0,2]
        hooks.stop=0
        x=torch.randn(1,3,8)
        with self.assertRaises(CaptureComplete):
            projection(x)
        self.assertTrue(torch.equal(hooks.features[0],x[0,[0,2]]))
        self.assertEqual(hooks.calls,{})
        hooks.close()
        projection(x)

    def test_tiny_llama_cache_and_sequential_inputs(self):
        config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=128,attention_dropout=0)
        config._attn_implementation='sdpa'
        model=LlamaForCausalLM(config).eval().requires_grad_(False)
        hooks=ProjectionHooks(model,[0,2])
        hooks.prefill=True
        hooks.capture={2}
        hooks.indices=[0,3]
        ids=torch.tensor([[1,3,7,9]])
        with torch.no_grad():
            original=model(ids).logits
            baseline=hooks.features[2].clone()
            hooks.enabled=True
            hooks.adapters={0:PreProjectionMLP(32,8),2:PreProjectionMLP(32,8)}
            cache=DynamicCache()
            same=model(ids,past_key_values=cache,use_cache=True).logits
            self.assertTrue(torch.equal(original,same))
            self.assertTrue(all(k.shape==(1,2,4,8) for k in cache.key_cache))
            calls=dict(hooks.calls)
            hooks.prefill=False
            model(torch.tensor([[4]]),past_key_values=cache,use_cache=True)
            self.assertEqual(calls,hooks.calls)
            self.assertTrue(all(k.shape==(1,2,5,8) for k in cache.key_cache))
            hooks.prefill=True
            hooks.adapters[0].up.bias.copy_(torch.linspace(-.2,.2,32))
            model(ids)
            self.assertFalse(torch.equal(baseline,hooks.features[2]))
        hooks.close()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_frozen_qonly_repeat_integration(self):
        from pilot import load, REPEAT
        layout=load('test_repeat_layout', REPEAT/'config.py')
        _, frozen, routing=layout.dependencies()
        cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=40,
            num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        cfg._attn_implementation='sdpa'
        model=LlamaForCausalLM(cfg).eval().requires_grad_(False)
        hooks=ProjectionHooks(model,layout.SPARSE_LAYERS)
        state=frozen.install(model,'repeat',routing.routed_attention,
            dense_layers=layout.DENSE_LAYERS,backend='sdpa',kv_mode='q_only')
        try:
            hooks.prefill=True
            ids=torch.tensor([[1,2,3,4,5,6,7]])
            def prefill():
                cache=DynamicCache()
                out=model(ids,past_key_values=cache,position_ids=torch.arange(7)[None],
                    attention_mask=torch.ones(1,7,dtype=torch.long),use_cache=True)
                return out.logits,cache
            with torch.no_grad():
                baseline,_=prefill()
                hooks.adapters={i:PreProjectionMLP(32,8) for i in layout.SPARSE_LAYERS}
                hooks.enabled=True
                actual,cache=prefill()
                self.assertTrue(torch.equal(actual,baseline))
                self.assertEqual(hooks.calls,{i:1 for i in layout.SPARSE_LAYERS})
                layout.validate_counts('repeat16',state['pairs'],7)
                hooks.prefill=False
                model(torch.tensor([[8]]),position_ids=torch.tensor([[7]]),
                    attention_mask=torch.ones(1,8,dtype=torch.long),past_key_values=cache,use_cache=True)
                self.assertEqual(hooks.calls,{i:1 for i in layout.SPARSE_LAYERS})
                self.assertTrue(all(k.shape==v.shape==(1,2,8,8) for k,v in zip(cache.key_cache,cache.value_cache)))
        finally:
            state['close']()
            hooks.close()


if __name__=='__main__':
    unittest.main()
