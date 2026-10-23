"""v5 decode path == v4 path for every hybrid variant, bitwise (outputs, pool, counters)."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvcompress.engine.w1_hybrid_variants_cache import HybridVariantLayer, mode_name, EXTRA
from kvcompress.engine.w1_hybrid_v5_cache import HybridV5Layer
from test_w1_hybrid_variants import make, step_inputs

STATE = ('cumulative_length', '_lengths', 'compaction_events', 'selection_events', 'selection_token_sum',
         'attention_token_sum', 'attention_token_observations', 'selected_pairs', 'causal_pairs',
         'max_selected_tokens', 'compaction_core_token_sum', 'compaction_residual_token_sum')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def compare(self, budget, prompt, interval, residual, n0, absolute, steps, shrink_at=None):
        mode = mode_name(interval, residual)
        old = make(HybridVariantLayer, mode, interval, n0, budget, absolute, prompt=prompt)
        new = make(HybridV5Layer, mode, interval, n0, budget, absolute, prompt=prompt)
        g = torch.Generator(device='cuda').manual_seed(29)
        rng_before = torch.cuda.get_rng_state().clone()
        for i in range(steps):
            if shrink_at is not None and i == shrink_at:
                for c in (old, new):
                    c.batch_select_indices(torch.tensor([1], device='cuda'))
            q, k, v = step_inputs(g, old.k_cache.shape[0])
            outs = []
            for c in (old, new):
                c.update_queries(q.transpose(1, 2))
                outs.append(c.attention(q, k, v, 128**-.5))
            label = (budget, prompt, interval, residual, i)
            self.assertTrue(torch.equal(*outs), label)
            self.assertTrue(torch.equal(old._ids, new._ids), label)
            n = new._lengths[0]
            self.assertTrue(torch.equal(old.slot_positions, new.slot_positions), label)
            self.assertTrue(torch.equal(old.k_cache[:, :n], new.k_cache[:, :n]), label)
            self.assertTrue(torch.equal(old.v_cache[:, :n], new.v_cache[:, :n]), label)
            self.assertTrue(torch.equal(old.cache_seqlens, new.cache_seqlens), label)
            self.assertEqual(old._rep_head, new._rep_head)
        for name in STATE:
            self.assertEqual(getattr(old, name), getattr(new, name), (name, label))
        self.assertTrue(torch.equal(rng_before, torch.cuda.get_rng_state()))
        return new

    def test_short_history_growth_past_budget(self):
        self.compare(1024, 137, 1000, 'random', 900, 900, 400)

    def test_all_variants_with_pool_updates_and_shrink(self):
        for budget in (1024, 2048):
            pool = budget+EXTRA
            for interval in (64, 1000, 1024):
                for residual in ('random', 'top'):
                    since = 10
                    aligned = interval*(-(-(pool+since)//interval)+1)
                    new = self.compare(budget, 137, interval, residual, pool+since, 137+aligned+since,
                                       2*interval+30 if interval == 64 else interval+60, shrink_at=40)
                    self.assertGreaterEqual(new.compaction_events, 1)

    def test_long_prompts(self):
        for budget, prompt in ((1024, 1567), (2048, 2792)):
            pool = budget+EXTRA
            for residual in ('random', 'top'):
                since = 10
                aligned = 64*(-(-(pool+since)//64)+1)
                new = self.compare(budget, prompt, 64, residual, pool+since, prompt+aligned+since, 160)
                self.assertGreaterEqual(new.compaction_events, 2)
                self.assertTrue(torch.equal(new.slot_positions[:, :prompt],
                                            torch.arange(prompt, device='cuda').expand(2, -1)))

    def test_tiny_qwen_generation_matches_v4_path(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_hybrid_variants_cache import install as install_v4_variants
        from kvcompress.engine.w1_hybrid_v5_cache import install as install_v5
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=8192, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        try:
            for prompt_len, interval, residual in ((5, 64, 'random'), (5, 64, 'top'), (5, 1000, 'random'),
                                                   (100, 64, 'random'), (100, 64, 'top')):
                inputs = torch.arange(1, prompt_len+1, device='cuda').remainder(127).add(1)[None].expand(2, -1).contiguous()
                outputs, metrics = {}, {}
                for name, install in (('v4', install_v4_variants), ('v5', install_v5)):
                    grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                    install('packed_qk_topk')
                    cc = dict(token_budget=80, residual_length=0, eviction_mode=mode_name(interval, residual),
                              rkv_lambda=0., smooth=False, verbose=False, top_p=.95, query_window=1,
                              eviction_interval=interval, rep_seed=0, sample_id=19)
                    torch.manual_seed(9182)
                    torch.cuda.manual_seed_all(9182)
                    metrics[name] = {}
                    outputs[name] = batch_exist_generate(model, input_ids=inputs,
                        attention_mask=torch.ones_like(inputs), max_length=4500, do_sample=True,
                        metrics=metrics[name], cache_implementation='evict', cache_config=cc)
                label = (prompt_len, interval, residual)
                self.assertTrue(torch.equal(outputs['v4'], outputs['v5']), label)
                for key in ('selected_pairs', 'causal_pairs', 'selection_events', 'attention_token_sum',
                            'max_selected_tokens'):
                    self.assertEqual(metrics['v4'][key], metrics['v5'][key], (key, label))
                self.assertIn('v5_fused_append', metrics['v5']['w1_implementation'])
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
