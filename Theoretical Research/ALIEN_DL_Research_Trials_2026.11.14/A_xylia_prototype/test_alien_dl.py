"""Fast mathematical and numerical smoke tests for the ALIEN-DL prototype."""
import unittest
import torch

from alien_dl import AlienConfig, AlienNet


class AlienNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = AlienNet(13, 3, AlienConfig())

    def test_resource_simplex_is_conserved(self):
        pi = self.model.allocation()
        self.assertTrue(torch.allclose(pi.sum(dim=-1), torch.full((8,), 13.0), atol=1e-6))
        self.assertTrue(torch.all(pi >= 0))

    def test_masked_synapses_receive_no_resource(self):
        with torch.no_grad():
            self.model.structural_mask[0, 0] = 0
        pi = self.model.allocation()
        self.assertAlmostEqual(float(pi[0, 0].detach()), 0.0, places=7)
        self.assertAlmostEqual(float(pi[0].sum().detach()), 12.0, places=5)

    def test_forward_backward_is_finite(self):
        x = torch.randn(16, 13)
        y = torch.randint(0, 3, (16,))
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (16, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(self.model.theta.grad).all())

    def test_allocation_update_preserves_simplex(self):
        utility = torch.randn(8, 13)
        self.model.allocation_step(utility)
        pi = self.model.allocation()
        self.assertTrue(torch.allclose(pi.sum(dim=-1), torch.full((8,), 13.0), atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
