"""v2/v3 equivalence: real GQA, short histories, ties, shrink and compaction."""
import unittest
import torch

from test_w1_qk_fast import layer
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer, FastHybridW1Layer
from kvcompress.engine.w1_qk_fused_cache import FusedFullW1Layer, FusedHybridW1Layer


@unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
class GPUChecks(unittest.TestCase):
    def test_selection_scores_ids_kv_lengths_and_rng(self):
        for old_cls, new_cls, budget, n in (
            (FastFullW1Layer, FusedFullW1Layer, 1024, 300),
            (FastFullW1Layer, FusedFullW1Layer, 1024, 1024),
            (FastFullW1Layer, FusedFullW1Layer, 1024, 1025),
            (FastFullW1Layer, FusedFullW1Layer, 1024, 32700),
            (FastFullW1Layer, FusedFullW1Layer, 2048, 8192),
            (FastFullW1Layer, FusedFullW1Layer, 201, 5120),
            (FastHybridW1Layer, FusedHybridW1Layer, 1024, 6120),
            (FastHybridW1Layer, FusedHybridW1Layer, 2048, 7144)):
            for backend in ('stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
                old = layer(old_cls, 'cuda', n, budget=budget, hkv=8, dim=128)
                new = layer(new_cls, 'cuda', n, budget=budget, hkv=8, dim=128)
                old.rank_backend = new.rank_backend = backend
                before_cpu = torch.random.get_rng_state().clone()
                before_gpu = torch.cuda.get_rng_state().clone()
                a, ac = old.select_indices()
                old.gather(a)
                b, bc = new.select_indices()
                new.gather(b)
                count = min(budget, n)
                self.assertTrue(torch.equal(a, b), (old_cls.__name__, budget, n, backend))
                self.assertEqual(ac, bc)
                self.assertEqual(old._rep_head, new._rep_head)
                self.assertTrue(torch.equal(old._last_scores, new._last_scores))
                self.assertTrue(torch.equal(old._temp_k[:, :count], new._temp_k[:, :count]))
                self.assertTrue(torch.equal(old._temp_v[:, :count], new._temp_v[:, :count]))
                self.assertEqual(new._selected_lengths.tolist(), bc)
                self.assertTrue(torch.equal(before_cpu, torch.random.get_rng_state()))
                self.assertTrue(torch.equal(before_gpu, torch.cuda.get_rng_state()))
                # Repeated gather must remain valid after the fused result is consumed.
                new.gather(b)
                self.assertTrue(torch.equal(old._temp_k[:, :count], new._temp_k[:, :count]))
                del old, new

    def test_attention_bitwise_equal_ties_short_history_and_batch_shrink(self):
        for old_cls, new_cls in ((FastFullW1Layer, FusedFullW1Layer),
                                  (FastHybridW1Layer, FusedHybridW1Layer)):
            for n in (300, 1024, 5800):
                old = layer(old_cls, 'cuda', n, hkv=8, dim=128)
                new = layer(new_cls, 'cuda', n, hkv=8, dim=128)
                for cache in (old, new):
                    cache.k_cache[:, :n].zero_()  # All QK scores tie.
                    cache.trace = False
                for shrink in (False, True):
                    if shrink:
                        ix = torch.tensor([1], device='cuda')
                        for cache in (old, new):
                            cache.batch_select_indices(ix)
                    q = old.queries.transpose(1, 2)
                    k = old.k_cache[:, n-1:n].contiguous()
                    v = old.v_cache[:, n-1:n].contiguous()
                    for cache in (old, new):
                        cache.cumulative_length = n
                        cache._lengths = [n]*q.shape[0]
                    a = old.attention(q, k, v, 128**-.5)
                    b = new.attention(q, k, v, 128**-.5)
                    self.assertTrue(torch.equal(a, b))
                    self.assertTrue(torch.equal(old._last_selected_ids, new._last_selected_ids))
                    self.assertEqual(old.selected_pairs, new.selected_pairs)
                    self.assertEqual(old.causal_pairs, new.causal_pairs)
                del old, new

    def test_hybrid_attention_and_two_physical_updates(self):
        for budget in (1024, 2048):
            n = budget+4096+999
            old = layer(FastHybridW1Layer, 'cuda', n, budget=budget, hkv=8, dim=128)
            new = layer(FusedHybridW1Layer, 'cuda', n, budget=budget, hkv=8, dim=128)
            for step in (10000, 11000):
                for cache in (old, new):
                    cache.trace = True
                    cache.cumulative_length = cache.prompt_len+step-1
                    cache._lengths = [n]*2
                    cache.cache_seqlens.fill_(n)
                q = old.queries.transpose(1, 2)
                k = old.k_cache[:, n-1:n].contiguous()
                v = old.v_cache[:, n-1:n].contiguous()
                before_gpu = torch.cuda.get_rng_state().clone()
                self.assertTrue(torch.equal(old.attention(q, k, v, 128**-.5),
                                            new.attention(q, k, v, 128**-.5)))
                self.assertTrue(torch.equal(before_gpu, torch.cuda.get_rng_state()))
                self.assertTrue(torch.equal(old.last_compaction_ids, new.last_compaction_ids))
                self.assertTrue(torch.equal(old.slot_positions, new.slot_positions))
                self.assertTrue(torch.equal(old.k_cache[:, :old.candidate_budget],
                                            new.k_cache[:, :new.candidate_budget]))
                self.assertTrue(torch.equal(old.v_cache[:, :old.candidate_budget],
                                            new.v_cache[:, :new.candidate_budget]))
                for rollout in old._generators:
                    self.assertTrue(torch.equal(old._generators[rollout].get_state(),
                                                new._generators[rollout].get_state()))
                # Refill identical new candidates for a second update.
                for cache in (old, new):
                    cache.k_cache[:, cache.candidate_budget:n].zero_()
                    cache.v_cache[:, cache.candidate_budget:n].zero_()
                    cache.slot_positions[:, cache.candidate_budget:n] = torch.arange(
                        cache.cumulative_length, cache.cumulative_length+999, device='cuda')
            self.assertEqual(new.compaction_events, 2)

    def test_tiny_qwen_sampled_generation(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_qk_fast_cache import install as v2_install
        from kvcompress.engine.w1_qk_fused_cache import install as v3_install
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=2048, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        inputs = torch.tensor([[1, 5, 11, 7, 23]]*2, device='cuda')
        try:
            for mode, interval, maximum in (
                ('grid_full_w1_topk_protected', 1, 165),
                ('grid_hybrid1000_w1_topk_protected', 1000, 1015)):
                cc = dict(token_budget=80, residual_length=0, eviction_mode=mode,
                          rkv_lambda=0., smooth=False, verbose=False, top_p=.95,
                          query_window=1, eviction_interval=interval, rep_seed=0, sample_id=19)
                outputs, states = [], []
                for install in (v2_install, v3_install):
                    grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                    install('packed_qk_topk')
                    torch.manual_seed(9182)
                    torch.cuda.manual_seed_all(9182)
                    outputs.append(batch_exist_generate(model, input_ids=inputs,
                        attention_mask=torch.ones_like(inputs), max_length=maximum, do_sample=True,
                        metrics={}, cache_implementation='evict', cache_config=cc))
                    states.append(torch.cuda.get_rng_state().clone())
                self.assertTrue(torch.equal(outputs[0], outputs[1]))
                self.assertTrue(torch.equal(states[0], states[1]))
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
