"""v4 hybrid: representative table, GPU chronological selection, masks, attention, generation."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvcompress.engine.cache_utils import _stable_hash
from kvcompress.engine.selector_grid_cache import representative
from kvcompress.engine.w1_qk_fast_cache import FastHybridW1Layer, exact_qk_topk
from kvcompress.engine.w1_hybrid_gpu_select_cache import (
    GpuSelectHybridW1Layer, stable_hash_steps, representative_table, gpu_select_chronological)
from test_w1_qk_fast import layer


class CPUChecks(unittest.TestCase):
    def test_vectorized_stable_hash_exact(self):
        rng = np.random.default_rng(3)
        steps = np.concatenate((np.arange(2000), rng.integers(0, 40000, 3000))).astype(np.uint64)
        for prefix in ((0x47524550, 0, 51, 0), (0x47524550, 0, 0, 35), (0x47524550, 7, 59, 63)):
            got = stable_hash_steps(prefix, steps)
            expected = [_stable_hash(*prefix, int(s)) for s in steps]
            self.assertEqual(got.tolist(), expected)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def test_representative_table_matches_private_stream(self):
        before = torch.random.get_rng_state().clone()
        for sample, hq in ((51, 32), (0, 32), (19, 8)):
            table = representative_table(0, sample, hq, 33000, 'cuda')
            for layer_idx in (0, 1, 17, 35, 63):
                for step in list(range(1, 1500)) + [4095, 4096, 9999, 32698, 32999]:
                    self.assertEqual(int(table[layer_idx, step]),
                                     representative(0, sample, layer_idx, step, hq), (sample, layer_idx, step))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_gpu_selection_equals_existing_ranking(self):
        g = torch.Generator(device='cuda').manual_seed(11)
        for n, start, count in ((1200, 137, 823), (4000, 70, 890), (4150, 70, 890),
                                (6120, 96, 864), (7144, 300, 1684), (5120, 17, 1943)):
            end = n-64
            for kind in ('normal', 'ties', 'zeros'):
                s = torch.randn(2, n, device='cuda', generator=g)
                if kind == 'ties':
                    s = (s*4).round()/4
                if kind == 'zeros':
                    s = torch.where(s > 0, torch.zeros_like(s), -torch.zeros_like(s))
                s = s.bfloat16()
                expected = exact_qk_topk(s, count, start, end, backend='packed_qk_topk')
                got = gpu_select_chronological(s, count, start, end)
                self.assertTrue(torch.equal(expected, got), (n, kind))
                stable = exact_qk_topk(s, count, start, end, backend='stable_sort')
                self.assertTrue(torch.equal(stable, got), (n, kind))

    def test_layer_masks_attention_compaction_and_shrink(self):
        # Pool length = candidate budget + steps since the last compaction (prompt 137).
        for budget, n0, absolute in ((1024, 5120+500, 137+6500), (2048, 6144+300, 137+9300),
                                     (1024, 900, 900)):
            old = layer(FastHybridW1Layer, 'cuda', n0, budget=budget, hkv=8, dim=128, absolute=absolute)
            new = layer(GpuSelectHybridW1Layer, 'cuda', n0, budget=budget, hkv=8, dim=128, absolute=absolute)
            old.trace = new.trace = False
            g = torch.Generator(device='cuda').manual_seed(5)
            cuda_before = torch.cuda.get_rng_state().clone()
            for i in range(2600):
                if i == 1300:
                    for cache in (old, new):
                        cache.batch_select_indices(torch.tensor([1], device='cuda'))
                b = old.k_cache.shape[0]
                q = torch.randn(b, 1, 32, 128, device='cuda', generator=g).bfloat16()
                k = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
                v = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
                outs = []
                for cache in (old, new):
                    cache.update_queries(q.transpose(1, 2))
                    outs.append(cache.attention(q, k, v, 128**-.5))
                self.assertTrue(torch.equal(*outs), (budget, i))
                self.assertTrue(torch.equal(old._ids, new._ids), (budget, i))
                self.assertEqual(old._rep_head, new._rep_head)
                self.assertTrue(torch.equal(old.slot_positions, new.slot_positions))
            self.assertEqual(old.compaction_events, new.compaction_events)
            if absolute > 5000:
                self.assertGreaterEqual(new.compaction_events, 2)
            self.assertEqual(old.selected_pairs, new.selected_pairs)
            self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state()))
            del old, new

    def test_tiny_qwen_sampled_generation_exact(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_qk_fast_cache import install as v2_install
        from kvcompress.engine.w1_hybrid_gpu_select_cache import install as v4_install
        from kvcompress.engine.w1_interval1000_cache import HYBRID_MODE
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=512, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        inputs = torch.tensor([[1, 5, 11, 7, 23]]*2, device='cuda')
        cc = dict(token_budget=80, residual_length=0, eviction_mode=HYBRID_MODE,
                  rkv_lambda=0., smooth=False, verbose=False, top_p=.95, query_window=1,
                  eviction_interval=1000, rep_seed=0, sample_id=19)
        outputs, metrics = {}, {}
        try:
            for name, install in (('v2', v2_install), ('v4', v4_install)):
                grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                install('packed_qk_topk')
                torch.manual_seed(9182)
                torch.cuda.manual_seed_all(9182)
                metrics[name] = {}
                outputs[name] = batch_exist_generate(model, input_ids=inputs,
                    attention_mask=torch.ones_like(inputs), max_length=420, do_sample=True,
                    metrics=metrics[name], cache_implementation='evict', cache_config=cc)
            self.assertTrue(torch.equal(outputs['v2'], outputs['v4']))
            self.assertEqual(metrics['v4']['w1_implementation'], 'hybrid_w1_raw_qk_gpu_select_v4')
            for key in ('selected_pairs', 'causal_pairs', 'selection_events', 'attention_token_sum'):
                self.assertEqual(metrics['v2'][key], metrics['v4'][key], key)
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
