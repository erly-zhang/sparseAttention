"""Pending approval: tests authored but not executed for the FA2 alignment."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from routing import routed_attention
from aligned_recorder import make_recorder


class AlignmentTests(unittest.TestCase):
    def test_grouped_flash_contract(self):
        # Mock validates API layout/masking only; real FA2 CUDA still required.
        torch.manual_seed(42)
        q = torch.randn(2, 4, 7, 8)
        k, v = torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
        valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]).bool()
        pos = (valid.long().cumsum(-1) - 1).clamp(min=0)
        calls = []
        def flash(qg, kg, vg, dropout_p, softmax_scale, causal):
            self.assertEqual(qg.shape, kg.shape)
            self.assertTrue(causal)
            self.assertEqual(dropout_p, 0)
            calls.append(qg.shape)
            return torch.nn.functional.scaled_dot_product_attention(
                qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2),
                is_causal=True, scale=softmax_scale).transpose(1, 2)
        fake = SimpleNamespace(flash_attn_func=flash)
        with patch.dict(sys.modules, {'flash_attn': fake}):
            for mode in ('q_only', 'q_kv'):
                expected, counts = routed_attention(q, k, v, pos, valid, mode, 8**-0.5)
                actual, actual_counts = routed_attention(q, k, v, pos, valid, mode, 8**-0.5,
                                                        backend='flash_attention_2', query_chunk=1)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(counts, actual_counts)
                if mode == 'q_only':
                    self.assertFalse(calls)
        self.assertEqual(len(calls), 8)

    def test_sidecar_outside_canonical_timer(self):
        order = []
        class Base:
            _installed = False
            def install(self):
                self._installed = True
                def timed(*args, **kwargs):
                    order.extend(['timer_start', 'generate', 'timer_stop', 'canonical_write'])
                    return 123
                self.model.generate = timed
        bridge = SimpleNamespace(rows=['warmup'], pending_row={'warmup': True},
                                 flush_pending=lambda: order.append('sidecar_write'))
        recorder = make_recorder(Base, lambda: bridge)()
        recorder.model = SimpleNamespace()
        recorder.install()
        self.assertEqual(bridge.rows, [])
        self.assertIsNone(bridge.pending_row)
        self.assertEqual(recorder.model.generate(), 123)
        self.assertEqual(order[-2:], ['canonical_write', 'sidecar_write'])


if __name__ == '__main__':
    unittest.main()
