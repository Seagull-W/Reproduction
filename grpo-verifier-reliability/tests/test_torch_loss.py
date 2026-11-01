import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch not installed")
class TorchLossTests(unittest.TestCase):
    def test_gradients_and_mask(self):
        from verifier_lab.torch_loss import grpo_loss

        current = torch.zeros(1, 2, 2, requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        loss = grpo_loss(
            current,
            old,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        )
        loss.backward()
        self.assertGreater(current.grad[0, 0, 0].item(), 0)
        self.assertLess(current.grad[0, 1, 0].item(), 0)
        self.assertEqual(current.grad[:, :, 1].abs().sum().item(), 0)
        self.assertIsNone(old.grad)

    def test_zero_advantage(self):
        from verifier_lab.torch_loss import grpo_loss

        x = torch.zeros(1, 2, 2, requires_grad=True)
        loss = grpo_loss(x, x.detach(), torch.ones(1, 2), torch.ones_like(x))
        loss.backward()
        self.assertEqual(x.grad.abs().sum().item(), 0)
