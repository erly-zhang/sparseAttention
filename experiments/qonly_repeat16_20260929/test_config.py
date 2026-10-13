import types
import unittest
from unittest.mock import patch
import config


class ConfigTests(unittest.TestCase):
    def test_alternating_layers(self):
        self.assertEqual(config.SPARSE_LAYERS, tuple(range(0, 32, 2)))
        self.assertEqual(config.DENSE_LAYERS, tuple(range(1, 32, 2)))
        self.assertEqual(set(config.SPARSE_LAYERS) | set(config.DENSE_LAYERS), set(range(32)))
        self.assertFalse(set(config.SPARSE_LAYERS) & set(config.DENSE_LAYERS))

    def test_explicit_route_mode_and_decode_contract(self):
        cfg = types.SimpleNamespace(num_hidden_layers=32, num_attention_heads=32,
                                    num_key_value_heads=8, hidden_size=4096)
        model = types.SimpleNamespace(config=cfg)
        with patch('config.dependencies') as deps:
            frozen = unittest.mock.Mock()
            route = types.SimpleNamespace(routed_attention=object())
            deps.return_value = (None, frozen, route)
            self.assertIsNone(config.install(model, 'dense'))
            deps.assert_not_called()
            for method, mode in config.MODES.items():
                config.install(model, method)
                frozen.install.assert_called_with(model, mode, route.routed_attention,
                    dense_layers=config.DENSE_LAYERS, backend='flash_attention_2', kv_mode='q_only')

    def test_reject_unknown_method_and_wrong_model(self):
        with self.assertRaises(ValueError):
            config.install(None, 'pair')
        model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=16,
            num_attention_heads=32, num_key_value_heads=8, hidden_size=4096))
        with self.assertRaises(ValueError):
            config.install(model, 'repeat16')

    def test_analytical_counts_and_wrong_old_layout(self):
        for n in (1, 7, 35, 130944):
            expected = {str(i): n * (n + 1) // 2 for i in config.SPARSE_LAYERS}
            for method in ('native16', 'repeat16'):
                config.validate_counts(method, expected, n)
            config.validate_counts('dense', {}, n)
            for wrong in ({}, {0: n}, {i: n for i in range(21)}):
                with self.assertRaises(ValueError):
                    config.validate_counts('repeat16', wrong, n)

    def test_protocol_has_no_mlp_and_no_rescaling(self):
        p = config.protocol()
        self.assertFalse(p['learned_adapter'])
        self.assertEqual(p['scale'], 1.0)
        self.assertTrue(p['prefill_only'] and p['dense_decode'] and p['full_kv_cache'])


try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
    TENSORS = True
except ImportError:
    TENSORS = False


@unittest.skipUnless(TENSORS, 'Requires the pinned PyTorch/Transformers CPU environment')
class TensorTests(unittest.TestCase):
    def test_qonly_merge_repeat_reference_and_causality(self):
        assembly, _, routing = config.dependencies()
        torch.manual_seed(42)
        h, hk, n, d = 4, 2, 9, 8
        q, k, v = torch.randn(1,h,n,d), torch.randn(1,hk,n,d), torch.randn(1,hk,n,d)
        pos = (torch.arange(n) * 3 + 1)[None]
        valid = torch.ones(1,n,dtype=torch.bool)
        expected = torch.zeros(1,n,d)
        for i in range(n):
            owner = int(pos[0,i]) % h
            a = (q[0,owner,i] @ k[0,owner//2,:i+1].T * d**-.5).softmax(-1)
            expected[0,i] = a @ v[0,owner//2,:i+1]
        merged, counts = routing.routed_attention(q,k,v,pos,valid,'q_only',d**-.5,backend='sdpa')
        torch.testing.assert_close(merged, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(counts['eligible_pairs'], n*(n+1)//2)
        repeated = assembly.assemble(merged,pos,valid,h,'repeat')
        for slot in range(h):
            torch.testing.assert_close(repeated.reshape(1,n,h,d)[:,:,slot],merged,rtol=0,atol=0)
        projection = torch.nn.Linear(h*d, 11, bias=True)
        torch.testing.assert_close(assembly.project(merged,pos,valid,projection,h,'repeat'),
                                   projection(expected.repeat(1,1,h)),atol=1e-6,rtol=1e-5)
        v[:,:,5:] += 50
        changed, _ = routing.routed_attention(q,k,v,pos,valid,'q_only',d**-.5,backend='sdpa')
        torch.testing.assert_close(changed[:,:5],merged[:,:5],rtol=0,atol=0)

    def test_exact16_layer_replacement_cache_and_dense_decode(self):
        _, frozen, routing = config.dependencies()
        cfg = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=40,
            num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        cfg._attn_implementation = 'sdpa'
        for mode in ('native','repeat'):
            model = LlamaForCausalLM(cfg).eval()
            originals = [layer.self_attn.forward for layer in model.model.layers]
            state = frozen.install(model,mode,routing.routed_attention,
                dense_layers=config.DENSE_LAYERS,backend='sdpa',kv_mode='q_only')
            try:
                changed = [i for i,layer in enumerate(model.model.layers)
                           if layer.self_attn.forward != originals[i]]
                self.assertEqual(changed,list(config.SPARSE_LAYERS))
                for n in (1,7):
                    cache = DynamicCache()
                    with torch.inference_mode():
                        out = model(input_ids=torch.arange(n)[None],position_ids=torch.arange(n)[None],
                            attention_mask=torch.ones(1,n,dtype=torch.long),past_key_values=cache,use_cache=True)
                        self.assertTrue(torch.isfinite(out.logits).all())
                        config.validate_counts('repeat16',state['pairs'],n)
                        for i in config.SPARSE_LAYERS:
                            def clone():
                                return DynamicCache.from_legacy_cache(tuple(
                                    (k.clone(),v.clone()) for k,v in cache.to_legacy_cache()))
                            x,pos = torch.randn(1,1,32),torch.tensor([[n]])
                            actual = model.model.layers[i].self_attn(x,position_ids=pos,past_key_value=clone(),use_cache=True)
                            expected = originals[i](x,position_ids=pos,past_key_value=clone(),use_cache=True)
                            torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
                        out = model(input_ids=torch.tensor([[2]]),position_ids=torch.tensor([[n]]),
                            attention_mask=torch.ones(1,n+1,dtype=torch.long),past_key_values=cache,use_cache=True)
                        self.assertTrue(torch.isfinite(out.logits).all())
                        self.assertEqual(len(cache.key_cache),32)
                        self.assertTrue(all(k.shape==v.shape==(1,2,n+1,8)
                            for k,v in zip(cache.key_cache,cache.value_cache)))
            finally:
                state['close']()
            self.assertEqual([l.self_attn.forward for l in model.model.layers], originals)


if __name__ == '__main__':
    if TENSORS:
        torch.set_num_threads(1)
    unittest.main(verbosity=2)
