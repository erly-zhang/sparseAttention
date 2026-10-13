import copy
import os
from pathlib import Path
import unittest
from protocol import CELLS, METHODS, code_indices, code_prompt, pair_count, validate, scorers


class ProtocolTests(unittest.TestCase):
    def test_selection_and_prompt(self):
        selected = code_indices(394)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 50)
        self.assertTrue(set(selected).isdisjoint(range(4)))
        self.assertEqual(selected, code_indices(394))
        row = dict(context='SOURCE', options=['fun1', 'fun2', 'fun3', 'fun4'], answer=['SECRET', 'A'])
        text = code_prompt(row)
        self.assertNotIn('SECRET', text)
        for value in ['SOURCE', 'A. fun1', 'B. fun2', 'C. fun3', 'D. fun4']:
            self.assertIn(value, text)

    def test_negative_identity_and_pair_validation(self):
        item = dict(task='code_debug', source_index=10, input_tokens=7,
                    input_ids_sha256='x', raw_row_sha256='y', answers=['fn', 'B'])
        rows = []
        for method in METHODS:
            counts = {} if method == 'dense' else {i: pair_count(7, CELLS[method][0]) for i in range(32)}
            rows.append(dict(**item, method=method, generated_ids=[2], generated_tokens=1,
                             full_kv_cache=True, dense_decode=True, finite_all_steps=True,
                             routed_layer_eligible_pairs=counts))
        self.assertTrue(validate(rows, [item], []))
        for field, value in [('input_ids_sha256', 'wrong'), ('finite_all_steps', False),
                             ('generated_tokens', 2), ('routed_layer_eligible_pairs', {0: 2})]:
            bad = copy.deepcopy(rows)
            bad[1][field] = value
            with self.assertRaises(AssertionError):
                validate(bad, [item], [])
        with self.assertRaises(AssertionError):
            validate(rows[:-1] + [rows[0]], [item], [])

    def test_pinned_official_scorers(self):
        local = Path(__file__).resolve().parents[2] / 'artifacts/psc_kl_quality/source_audit/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill/experiments/benchmark/infinitebench/metrics.py'
        scores = scorers(Path(os.environ.get('METRICS_SOURCE', local)))
        self.assertEqual(scores['kv_retrieval'](('abc', '"abc"')), 1)
        self.assertEqual(scores['kv_retrieval'](('abc', 'abcd')), 0)
        self.assertEqual(scores['code_debug'](('fun===answerspliter===B', 'B. fun')), 1)
        self.assertEqual(scores['code_debug'](('fun===answerspliter===B', 'A. wrong')), 0)


class TensorTests(unittest.TestCase):
    def test_native_slot_bias_padding_and_original_positions(self):
        import torch
        from assembly import assemble, project
        torch.manual_seed(42)
        x = torch.randn(2, 7, 8, dtype=torch.float64)
        pos = torch.tensor([[1, 3, 4, 8, 9, 12, 15]]).expand(2, -1)
        valid = torch.ones_like(pos, dtype=torch.bool)
        valid[1, -1] = False
        expected = torch.zeros(2, 7, 4, 8, dtype=x.dtype)
        for b in range(2):
            for i in range(7):
                if valid[b, i]:
                    expected[b, i, int(pos[b, i]) % 4] = x[b, i]
        torch.testing.assert_close(assemble(x, pos, valid, 4, 'native'), expected.flatten(-2), rtol=0, atol=0)
        linear = torch.nn.Linear(32, 11, bias=True).double()
        torch.testing.assert_close(project(x, pos, valid, linear, 4, 'native'), linear(expected.flatten(-2)))

    def test_both_routes_independent_reference_and_causality(self):
        import torch
        from test_assembly import load_routing
        route = load_routing().routed_attention
        torch.manual_seed(12)
        for n in (1, 3, 7, 35):
            q, k, v = torch.randn(2, 4, n, 8), torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
            pos = (torch.arange(n) * 3 + 7)[None].expand(2, -1)
            valid = torch.ones(2, n, dtype=torch.bool)
            if n > 1:
                valid[1, -1] = False
            for mode in ('q_kv', 'q_only'):
                expected = torch.zeros(2, n, 8)
                count = 0
                for b in range(2):
                    for i in range(n):
                        if not valid[b, i]:
                            continue
                        h = int(pos[b, i]) % 4
                        keep = valid[b] & (pos[b] <= pos[b, i])
                        if mode == 'q_kv':
                            keep &= pos[b] % 4 == h
                        count += int(keep.sum())
                        scores = (q[b, h, i] @ k[b, h // 2, keep].T * 8**-.5).softmax(-1)
                        expected[b, i] = scores @ v[b, h // 2, keep]
                for backend in ('eager', 'sdpa'):
                    actual, counts = route(q, k, v, pos, valid, mode, 8**-.5, backend=backend, query_chunk=2)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(counts['eligible_pairs'], count)
                    cutoff = max(1, n // 2)
                    k2, v2 = k.clone(), v.clone()
                    k2[:, :, cutoff:] += 50
                    v2[:, :, cutoff:] -= 30
                    changed, _ = route(q, k2, v2, pos, valid, mode, 8**-.5, backend=backend, query_chunk=2)
                    torch.testing.assert_close(actual[:, :cutoff], changed[:, :cutoff], atol=0, rtol=0)

    def test_all_four_adapters_dense_decode_and_full_cache(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
        from adapter import install
        from test_assembly import load_routing
        config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        for kv_mode, assembly in CELLS.values():
            model = LlamaForCausalLM(config).eval()
            original = model.model.layers[0].self_attn.forward
            state = install(model, assembly, load_routing().routed_attention,
                            dense_layers=(1,), backend='sdpa', kv_mode=kv_mode)
            for n in (1, 7):
                cache = DynamicCache()
                with torch.inference_mode():
                    out = model(input_ids=torch.arange(n)[None], position_ids=torch.arange(n)[None],
                        attention_mask=torch.ones(1, n, dtype=torch.long), past_key_values=cache, use_cache=True)
                    self.assertTrue(torch.isfinite(out.logits).all())
                    self.assertEqual(state['pairs'][0], pair_count(n, kv_mode, 4))
                    for j in range(2):
                        clone = lambda: DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in cache.to_legacy_cache()))
                        x, pos = torch.randn(1, 1, 32), torch.tensor([[n+j]])
                        actual = model.model.layers[0].self_attn(x, position_ids=pos, past_key_value=clone(), use_cache=True)
                        expected = original(x, position_ids=pos, past_key_value=clone(), use_cache=True)
                        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                        out = model(input_ids=torch.tensor([[2]]), position_ids=pos,
                            attention_mask=torch.ones(1, n+j+1, dtype=torch.long), past_key_values=cache, use_cache=True)
                        self.assertTrue(torch.isfinite(out.logits).all())
                    self.assertTrue(all(k.shape == v.shape == (1, 2, n+2, 8)
                        for k, v in zip(cache.key_cache, cache.value_cache)))
            state['close']()
            self.assertEqual(model.model.layers[0].self_attn.forward, original)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
