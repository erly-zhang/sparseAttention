"""Complete sampled token trajectory through the same Qwen3 runner (tiny model)."""
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from kvcompress.engine.cache_utils import EvictCache
from kvcompress.engine.modify_forward import wrap_evict_attn_forward
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer, install
import kvcompress.engine.selector_grid_cache as grid
from kvcompress.harness.generation_utils import batch_exist_generate


class FullW1ModelTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
    def test_complete_sampled_token_ids_rope_gqa_and_factory(self):
        original_class=grid.SelectorGridLayer
        original_forward=Qwen3Attention.forward
        config=Qwen3Config(vocab_size=128,hidden_size=128,intermediate_size=256,
                          num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=2,head_dim=16,
                          max_position_embeddings=256,eos_token_id=128,pad_token_id=0)
        config._attn_implementation='flash_attention_2'
        torch.manual_seed(914)
        model=Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
        model.generation_config.temperature=.6;model.generation_config.top_p=.95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        ids=torch.tensor([[1,5,11,7,23],[1,5,11,7,23]],device='cuda')
        cache_config=dict(token_budget=80,residual_length=0,eviction_mode='grid_full_w1_topk_protected',
                          rkv_lambda=0.,smooth=False,verbose=False,top_p=.95,query_window=1,
                          eviction_interval=1,rep_seed=0,sample_id=19)
        outputs={};metrics={}
        try:
            for implementation in ['baseline','stable_sort','packed_topk']:
                grid.SelectorGridLayer=original_class
                if implementation!='baseline':
                    install(implementation)
                    cache=EvictCache(model.config,**cache_config)
                    self.assertTrue(all(type(layer) is FullW1OptimizedLayer for layer in cache.layers))
                torch.manual_seed(9182);torch.cuda.manual_seed_all(9182)
                m={}
                outputs[implementation]=batch_exist_generate(model,input_ids=ids,attention_mask=torch.ones_like(ids),
                    max_length=165,do_sample=True,metrics=m,cache_implementation='evict',cache_config=cache_config)
                metrics[implementation]=m
            for implementation in ['stable_sort','packed_topk']:
                self.assertTrue(torch.equal(outputs['baseline'],outputs[implementation]))
                for key in ['generated_tokens_per_rollout','eos_reached','selection_events','selected_pairs','causal_pairs',
                            'final_cache_seqlens','final_cumulative_length','per_layer_selected_pairs','per_layer_causal_pairs']:
                    self.assertEqual(metrics['baseline'][key],metrics[implementation][key],key)
        finally:
            grid.SelectorGridLayer=original_class
            Qwen3Attention.forward=original_forward


if __name__=='__main__':
    unittest.main(verbosity=2)
