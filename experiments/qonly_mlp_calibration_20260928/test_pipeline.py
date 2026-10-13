import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required; remote tests are mandatory')
class PipelineTests(unittest.TestCase):
    def test_aligned_teacher_prefix_identity_and_restore(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        import pilot
        from calibrator import ResidualReadout
        pilot.load('assembly', pilot.ABLATION / 'assembly.py')
        adapter = pilot.load('test_qonly_adapter', pilot.ABLATION / 'adapter.py')
        route = pilot.load('test_qonly_route', pilot.ROUTING).routed_attention
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
            eos_token_id=2)).eval().requires_grad_(False)

        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return ' '.join(map(str, ids))

        engine = pilot.Engine(model, Tokenizer(), adapter.install, route, (1,), 'cpu')
        original = model.model.layers[0].self_attn.forward
        old_limit = pilot.MAX_NEW
        pilot.MAX_NEW = 4
        try:
            identity = ResidualReadout(32, 3).eval()
            for ids in ([1], [1, 3, 4, 5, 6, 7, 8]):
                dense, dh, _ = engine.generate(ids, 'dense')
                replay, sh, _ = engine.generate(ids, teacher_ids=dense['generated_ids'])
                self.assertEqual(dense['generated_ids'], replay['generated_ids'])
                self.assertEqual(dh.shape, sh.shape)
                self.assertEqual(replay['routed_layer_eligible_pairs'], {0: len(ids) * (len(ids) + 1) // 2})
                plain, ph, pl = engine.generate(ids)
                initial, ih, il = engine.generate(ids, adapter=identity)
                self.assertEqual(plain['generated_ids'], initial['generated_ids'])
                torch.testing.assert_close(ph, ih, rtol=0, atol=0)
                self.assertTrue((pl == il).all())
                self.assertEqual(model.model.layers[0].self_attn.forward, original)
            with self.assertRaises(AssertionError):
                engine.generate([1], teacher_ids=[2, 3])
        finally:
            pilot.MAX_NEW = old_limit

    def test_checkpoint_round_trip(self):
        import io
        from calibrator import ResidualReadout
        module = ResidualReadout(8, 3)
        with torch.no_grad():
            module.up.weight.normal_()
        stream = io.BytesIO()
        torch.save(module.state_dict(), stream)
        stream.seek(0)
        restored = ResidualReadout(8, 3)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        x = torch.randn(3, 8)
        torch.testing.assert_close(module(x, enabled=True), restored(x, enabled=True), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
