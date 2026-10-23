"""v6 on Phi-4-reasoning geometry (40 Q / 10 KV heads, KV width 1280, not a power of two)."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvcompress.engine.w1_hybrid_variants_cache import HybridVariantLayer, mode_name, EXTRA
from kvcompress.engine.w1_hybrid_v6_cache import HybridV6Layer

HQ, HKV, D = 40, 10, 128


def make(cls, mode, interval, n, budget, absolute, prompt=300, batch=2):
    layer = cls(5, budget, 0, mode, 0., False, False, query_window=1, eviction_interval=interval, rep_seed=0,
                sample_id=11, max_cache_length=32768, track_positions=False)
    layer.lazy_initialization(torch.zeros(batch, prompt, HKV, D, device='cuda', dtype=torch.bfloat16))
    g = torch.Generator(device='cuda').manual_seed(932)
    layer.k_cache[:, :n].normal_(generator=g)
    layer.v_cache[:, :n].normal_(generator=g)
    layer.slot_positions[:, :n] = torch.cat((torch.arange(prompt, device='cuda'),
                                             torch.arange(absolute-(n-prompt), absolute, device='cuda')))
    layer.cumulative_length = absolute
    layer._lengths = [n]*batch
    layer.cache_seqlens.fill_(n)
    layer._num_query_heads = HQ
    return layer


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PhiGeometry(unittest.TestCase):
    def compare(self, budget, prompt, interval, n0, absolute, steps):
        mode = mode_name(interval, 'random')
        old = make(HybridVariantLayer, mode, interval, n0, budget, absolute, prompt)
        new = make(HybridV6Layer, mode, interval, n0, budget, absolute, prompt)
        g = torch.Generator(device='cuda').manual_seed(4)
        for i in range(steps):
            if i == steps//2:
                for c in (old, new):
                    c.batch_select_indices(torch.tensor([1], device='cuda'))
            b = old.k_cache.shape[0]
            q = torch.randn(b, 1, HQ, D, device='cuda', generator=g).bfloat16()
            k = torch.randn(b, 1, HKV, D, device='cuda', generator=g).bfloat16()
            v = torch.randn(b, 1, HKV, D, device='cuda', generator=g).bfloat16()
            outs = []
            for c in (old, new):
                c.update_queries(q.transpose(1, 2))
                outs.append(c.attention(q, k, v, D**-.5))
            self.assertTrue(torch.equal(*outs), (budget, prompt, i))
            n = new._lengths[0]
            self.assertTrue(torch.equal(old._ids, new._ids), i)
            self.assertTrue(torch.equal(old.k_cache[:, :n], new.k_cache[:, :n]), i)
            self.assertTrue(torch.equal(old.v_cache[:, :n], new.v_cache[:, :n]), i)
            self.assertTrue(torch.equal(old.slot_positions, new.slot_positions), i)
        for name in ('selected_pairs', 'compaction_events', 'selection_events', 'attention_token_sum'):
            self.assertEqual(getattr(old, name), getattr(new, name), name)
        return new

    def test_pool_updates_and_shrink(self):
        for budget in (1024, 2048):
            pool = budget+EXTRA
            since = 10
            aligned = 1024*(-(-(pool+since)//1024)+1)
            new = self.compare(budget, 300, 1024, pool+since, 300+aligned+since, 1100)
            self.assertGreaterEqual(new.compaction_events, 1)

    def test_long_prompt_and_short_history(self):
        new = self.compare(1024, 1500, 1024, 1600, 1600, 300)    # prompt > K-64
        self.assertEqual(new._protect_len, 1024-65)
        self.compare(2048, 300, 1024, 900, 900, 1300)             # grows past K

    def test_tiny_phi3_generation_matches_v4_path(self):
        from transformers import Phi3Config, Phi3ForCausalLM
        from transformers.models.phi3.modeling_phi3 import Phi3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_hybrid_variants_cache import install as install_v4
        from kvcompress.engine.w1_hybrid_v6_cache import install as install_v6
        original = grid.SelectorGridLayer, grid.decode_mode
        forward = Phi3Attention.forward
        config = Phi3Config(vocab_size=128, hidden_size=160, intermediate_size=256, num_hidden_layers=2,
                            num_attention_heads=10, num_key_value_heads=5, max_position_embeddings=8192,
                            eos_token_id=127, pad_token_id=0, bos_token_id=1)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(13)
        model = Phi3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .8
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('tiny-phi-4-reasoning')
        try:
            for prompt_len in (5, 100):
                inputs = torch.arange(2, prompt_len+2, device='cuda')[None].expand(2, -1).contiguous()
                outs, metrics = {}, {}
                for name, install in (('v4', install_v4), ('v6', install_v6)):
                    grid.SelectorGridLayer, grid.decode_mode = original
                    install('packed_qk_topk')
                    cc = dict(token_budget=80, residual_length=0, eviction_mode=mode_name(1024, 'random'),
                              rkv_lambda=0., smooth=False, verbose=False, top_p=.95, query_window=1,
                              eviction_interval=1024, rep_seed=0, sample_id=19)
                    torch.manual_seed(91)
                    torch.cuda.manual_seed_all(91)
                    metrics[name] = {}
                    outs[name] = batch_exist_generate(model, input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                                      max_length=600, do_sample=True, metrics=metrics[name],
                                                      cache_implementation='evict', cache_config=cc)
                self.assertTrue(torch.equal(outs['v4'], outs['v6']), prompt_len)
                for key in ('selected_pairs', 'causal_pairs', 'selection_events'):
                    self.assertEqual(metrics['v4'][key], metrics['v6'][key], (key, prompt_len))
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original
            Phi3Attention.forward = forward


if __name__ == '__main__':
    unittest.main(verbosity=2)
