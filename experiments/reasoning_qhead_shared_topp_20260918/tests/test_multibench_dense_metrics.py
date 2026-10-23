"""Dense timing/metric branch of batch_exist_generate (standard DynamicCache)."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DenseMetrics(unittest.TestCase):
    def test_dense_metrics_and_exact_pair_counts(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from kvcompress.harness.generation_utils import batch_exist_generate
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=1024, eos_token_id=7, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(5)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        inputs = torch.tensor([[1, 5, 11, 9, 23, 4]]*2, device='cuda')
        metrics = {}
        torch.manual_seed(31)
        torch.cuda.manual_seed_all(31)
        out = batch_exist_generate(model, input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                   max_length=300, do_sample=True, metrics=metrics)
        P = inputs.shape[1]
        ts = metrics['generated_tokens_per_rollout']
        factor = 2*8
        self.assertEqual(metrics['cache_semantics'], 'standard_dense_DynamicCache')
        self.assertEqual(metrics['selection_events'], 0)
        self.assertEqual(metrics['selected_pairs'], factor*sum((P+t-1)*(P+t)//2 for t in ts))
        self.assertEqual(metrics['selected_pairs'], metrics['causal_pairs'])
        self.assertEqual(metrics['prefill_selected_pairs'], 2*factor*P*(P+1)//2)
        self.assertEqual(metrics['decode_forward_calls'], max(ts)-1)
        self.assertEqual(metrics['decode_query_tokens'], sum(ts)-2)
        self.assertEqual(metrics['final_cache_length'], P+max(ts)-1)
        self.assertGreater(metrics['decode_seconds'], 0)
        self.assertGreater(metrics['prefill_seconds'], 0)
        self.assertEqual(out.shape[0], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
