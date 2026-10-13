import unittest

try:
    import torch
    from calibrator import ResidualReadout, paired_loss
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required; do not treat skipped tests as PASS')
class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.adapter = ResidualReadout(8, 3)
        self.head = torch.nn.Linear(8, 17, bias=False).requires_grad_(False)
        self.sparse = torch.randn(4, 8)
        self.dense = torch.randn(4, 8)

    def test_identity_including_first_answer_and_decode_shapes(self):
        for shape in ((1, 8), (4, 8), (1, 1, 8)):
            for dtype in (torch.float32, torch.bfloat16):
                h = torch.randn(shape).to(dtype)
                self.assertTrue(torch.equal(h, self.adapter(h, enabled=True)))
                self.assertEqual(h.dtype, self.adapter(h, enabled=True).dtype)

    def test_dense_bypass_even_after_training(self):
        with torch.no_grad():
            self.adapter.up.bias.fill_(1)
        self.assertIs(self.adapter(self.sparse, enabled=False), self.sparse)

    def test_frozen_teacher_and_head_student_gradient(self):
        sparse = self.sparse.requires_grad_()
        dense = self.dense.requires_grad_()
        loss = paired_loss(self.adapter, self.head, sparse, dense, alpha=0)
        loss.backward()
        self.assertGreater(self.adapter.up.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.head.weight.grad)
        self.assertIsNone(sparse.grad)
        self.assertIsNone(dense.grad)
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()
        paired_loss(self.adapter, self.head, sparse, dense).backward()
        self.assertGreater(self.adapter.down.weight.grad.abs().sum().item(), 0)

    def test_equal_features_zero_initial_loss(self):
        loss = paired_loss(self.adapter, self.head, self.sparse, self.sparse)
        self.assertAlmostEqual(loss.item(), 0, places=6)

    def test_weighted_positions(self):
        weights = torch.tensor([1., 0., 0., 0.])
        a = paired_loss(self.adapter, self.head, self.sparse, self.dense, weights=weights)
        b = paired_loss(self.adapter, self.head, self.sparse[:1], self.dense[:1])
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_reject_unfrozen_head_and_bad_weights(self):
        self.head.requires_grad_(True)
        with self.assertRaises(ValueError):
            paired_loss(self.adapter, self.head, self.sparse, self.dense)
        self.head.requires_grad_(False)
        for weights in (torch.zeros(4), torch.ones(3), -torch.ones(4)):
            with self.assertRaises(ValueError):
                paired_loss(self.adapter, self.head, self.sparse, self.dense, weights=weights)

    def test_linear_control_same_parameter_budget(self):
        linear = ResidualReadout(8, 3, nonlinear=False)
        self.assertEqual(sum(p.numel() for p in linear.parameters()),
                         sum(p.numel() for p in self.adapter.parameters()))


if __name__ == '__main__':
    unittest.main()
