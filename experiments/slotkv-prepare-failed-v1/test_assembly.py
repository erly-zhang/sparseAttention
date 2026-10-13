import importlib.util
import os
from pathlib import Path
import unittest
import torch
from assembly import assemble, head_pairs, project


def load_routing():
    default = Path(__file__).resolve().parents[2] / "artifacts/psc_kl_quality/source_audit/experiments/prefill_token_head_routing_20260924/routing.py"
    path = Path(os.environ.get("ROUTING_SOURCE", default))
    spec = importlib.util.spec_from_file_location("frozen_test_routing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssemblyTests(unittest.TestCase):
    def test_exact_three_head_example(self):
        values = torch.arange(1., 7.).view(1, 6, 1)
        # Explicit owners 0,0,1,1,2,2; production modulo routing is unchanged.
        pos = torch.tensor([[0, 3, 4, 7, 8, 11]])
        actual = assemble(values, pos, torch.ones_like(pos, dtype=torch.bool), 3, "pair")
        expected = torch.tensor([[[1, 1, 0], [2, 2, 0], [3, 0, 3],
                                  [4, 0, 4], [0, 5, 5], [0, 6, 6]]], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_balanced_pairs(self):
        self.assertEqual(head_pairs(3), ((0, 1), (0, 2), (1, 2)))
        for h in (3, 4, 16, 32):
            pairs = head_pairs(h)
            self.assertEqual(len(set(pairs)), h)
            self.assertEqual([sum(i in p for p in pairs) for i in range(h)], [2]*h)
        with self.assertRaises(ValueError):
            head_pairs(2)

    def test_padding_short_nondivisible_and_bias(self):
        for h in (3, 16, 32):
            for n in (1, 2, 17, 35):
                x = torch.randn(2, n, 4, dtype=torch.float64)
                pos = torch.arange(n)[None].expand(2, -1) + 7
                valid = torch.ones(2, n, dtype=torch.bool)
                valid[1, -1] = False
                out = assemble(x, pos, valid, h, "pair").view(2, n, h, 4)
                for b in range(2):
                    for i in range(n):
                        for c, p in enumerate(head_pairs(h)):
                            expected = x[b, i] if valid[b, i] and int(pos[b, i]) % h in p else torch.zeros(4, dtype=x.dtype)
                            torch.testing.assert_close(out[b, i, c], expected, rtol=0, atol=0)
                linear = torch.nn.Linear(h*4, 7, bias=True).double()
                actual = project(x, pos, valid, linear, h, "pair")
                reference = torch.zeros(2, n, 7, dtype=x.dtype)
                for c, p in enumerate(head_pairs(h)):
                    take = valid & ((pos % h == p[0]) | (pos % h == p[1]))
                    reference += (x * take[..., None]) @ linear.weight[:, c*4:(c+1)*4].T
                reference += linear.bias
                torch.testing.assert_close(actual, reference)

    def test_repeat_regression(self):
        old = load_routing()
        x = torch.randn(1, 11, 4)
        pos = torch.arange(11)[None]
        linear = torch.nn.Linear(16, 9)
        actual = project(x, pos, torch.ones_like(pos, dtype=torch.bool), linear, 4, "repeat")
        torch.testing.assert_close(actual, old.repeat_project(x, linear, 4), rtol=0, atol=0)

    def test_causal_gqa_unchanged_pairs(self):
        route = load_routing().routed_attention
        torch.manual_seed(42)
        for n in (1, 2, 7, 17):
            q, k, v = torch.randn(1, 4, n, 8), torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8)
            pos = torch.arange(n)[None]
            valid = torch.ones_like(pos, dtype=torch.bool)
            m, counts = route(q, k, v, pos, valid, "q_kv", 8**-0.5)
            ref = torch.zeros_like(m)
            for i in range(n):
                h = i % 4
                keys = torch.arange(h, i+1, 4)
                weights = (q[0, h, i] @ k[0, h//2, keys].T / 8**0.5).softmax(-1)
                ref[0, i] = weights @ v[0, h//2, keys]
            torch.testing.assert_close(m, ref)
            prefix = max(1, n//2)
            q[:, :, prefix:] += 100
            k[:, :, prefix:] -= 100
            v[:, :, prefix:] += 200
            changed, counts2 = route(q, k, v, pos, valid, "q_kv", 8**-0.5)
            for mode in ("repeat", "pair"):
                torch.testing.assert_close(assemble(m, pos, valid, 4, mode)[:, :prefix],
                    assemble(changed, pos, valid, 4, mode)[:, :prefix], rtol=0, atol=0)
            self.assertEqual(counts, counts2)

    def test_tiny_llama_dense_decode_and_cache(self):
        from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
        from adapter import install
        route = load_routing().routed_attention
        config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.0)
        for mode in ("repeat", "pair"):
            model = LlamaForCausalLM(config).eval()
            original = model.model.layers[0].self_attn.forward
            state = install(model, mode, route, dense_layers=(1,), backend="eager")
            for n in (1, 7):
                cache = DynamicCache()
                with torch.inference_mode():
                    out = model(input_ids=torch.arange(n)[None], position_ids=torch.arange(n)[None],
                        attention_mask=torch.ones(1, n, dtype=torch.long), past_key_values=cache, use_cache=True)
                    self.assertTrue(torch.isfinite(out.logits).all())
                    for j in range(2):
                        layer = model.model.layers[0].self_attn
                        x = torch.randn(1, 1, 32)
                        pos = torch.tensor([[n+j]])
                        a = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in cache.to_legacy_cache()))
                        b = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in cache.to_legacy_cache()))
                        actual = layer(x, position_ids=pos, past_key_value=a, use_cache=True)
                        expected = original(x, position_ids=pos, past_key_value=b, use_cache=True)
                        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                        out = model(input_ids=torch.tensor([[2]]), position_ids=pos,
                            attention_mask=torch.ones(1, n+j+1, dtype=torch.long), past_key_values=cache, use_cache=True)
                        self.assertTrue(torch.isfinite(out.logits).all())
                    self.assertEqual(cache.get_seq_length(), n+2)
                    self.assertEqual(tuple(cache.key_cache[0].shape), (1, 2, n+2, 8))
            state["close"]()
            self.assertEqual(model.model.layers[0].self_attn.forward, original)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
