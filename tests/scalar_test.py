import unittest
import torch
from drop import scalar

class TestScalarAutograd(unittest.TestCase):
  
  def setUp(self):
    # Setting up initial values for PyTorch tensors
    self.x1_torch = torch.tensor([2.0], requires_grad=True)
    self.x2_torch = torch.tensor([3.0], requires_grad=True)
    self.x3_torch = torch.tensor([5.0], requires_grad=True)
    self.x4_torch = torch.tensor([10.0], requires_grad=True)
    self.x5_torch = torch.tensor([1.0], requires_grad=True)
    self.x6_torch = torch.tensor([4.0], requires_grad=True)
    self.x7_torch = torch.tensor([-2.0], requires_grad=True)

    # Setting up initial values for drop.scalars
    self.x1_drop = scalar(2)
    self.x2_drop = scalar(3)
    self.x3_drop = scalar(5)
    self.x4_drop = scalar(10)
    self.x5_drop = scalar(1)
    self.x6_drop = scalar(4)
    self.x7_drop = scalar(-2)

  def test_forward_and_backward(self):
    # --- PyTorch Computation ---
    a1_torch = self.x1_torch + self.x2_torch
    a2_torch = self.x3_torch - self.x4_torch
    a3_torch = a1_torch * a2_torch
    a4_torch = a3_torch ** 2
    a5_torch = self.x5_torch * self.x6_torch
    a6_torch = a5_torch.sigmoid()
    a7_torch = self.x7_torch.tanh()
    a8_torch = a4_torch + a6_torch
    a9_torch = a8_torch + a7_torch
    y_torch = a9_torch.relu()

    # Retaining gradients for PyTorch intermediate values
    a1_torch.retain_grad()
    a2_torch.retain_grad()
    a3_torch.retain_grad()
    a4_torch.retain_grad()
    a5_torch.retain_grad()
    a6_torch.retain_grad()
    a7_torch.retain_grad()
    a8_torch.retain_grad()
    a9_torch.retain_grad()
    y_torch.retain_grad()
    
    y_torch.backward()
    
    # --- Drop Scalar Computation ---
    a1_drop = self.x1_drop + self.x2_drop
    a2_drop = self.x3_drop - self.x4_drop
    a3_drop = a1_drop * a2_drop
    a4_drop = a3_drop ** 2
    a5_drop = self.x5_drop * self.x6_drop
    a6_drop = a5_drop.sigmoid()
    a7_drop = self.x7_drop.tanh()
    a8_drop = a4_drop + a6_drop
    a9_drop = a8_drop + a7_drop
    y_drop = a9_drop.relu()

    y_drop.backward()

    # --- Assertions ---
    # 1. Compare forward outputs
    self.assertAlmostEqual(y_torch.item(), y_drop.item(), places=5, msg="Forward outputs do not match!")

    # 2. Compare gradients
    self.assertAlmostEqual(self.x1_torch.grad.item(), self.x1_drop.grad, places=5, msg="Gradient mismatch for x1")
    self.assertAlmostEqual(self.x2_torch.grad.item(), self.x2_drop.grad, places=5, msg="Gradient mismatch for x2")
    self.assertAlmostEqual(self.x3_torch.grad.item(), self.x3_drop.grad, places=5, msg="Gradient mismatch for x3")
    self.assertAlmostEqual(self.x4_torch.grad.item(), self.x4_drop.grad, places=5, msg="Gradient mismatch for x4")
    self.assertAlmostEqual(self.x5_torch.grad.item(), self.x5_drop.grad, places=5, msg="Gradient mismatch for x5")
    self.assertAlmostEqual(self.x6_torch.grad.item(), self.x6_drop.grad, places=5, msg="Gradient mismatch for x6")
    self.assertAlmostEqual(self.x7_torch.grad.item(), self.x7_drop.grad, places=5, msg="Gradient mismatch for x7")

if __name__ == '__main__':
  unittest.main()