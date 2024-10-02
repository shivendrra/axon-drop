import unittest
import torch
from drop import tensor
import numpy as np

class TestDropTensorAutograd(unittest.TestCase):
  
  def setUp(self):
    # Setting up initial values for drop.tensor
    self.a_drop = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]])
    self.b_drop = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]])

    # Setting up initial values for PyTorch tensors
    self.a_torch = torch.tensor([[2.0, 4.0, 5.0, -4.0], [-3.0, 0.0, 9.0, -1.0]], requires_grad=True)
    self.b_torch = torch.tensor([[1.0, 0.0, -2.0, 0.0], [-1.0, 10.0, -2.0, 4.0]], requires_grad=True)

  def test_forward_and_backward(self):
    # --- Drop Tensor Computation ---
    c_drop = self.a_drop + self.b_drop
    d_drop = c_drop.tanh()
    e_drop = d_drop.silu()
    f_drop = e_drop ** 2
    g_drop = f_drop.sigmoid()
    h_drop = g_drop.sum()
    
    h_drop.backward()

    # --- PyTorch Computation ---
    c_torch = self.a_torch + self.b_torch
    d_torch = torch.tanh(c_torch)
    e_torch = torch.nn.functional.silu(d_torch)
    f_torch = e_torch ** 2
    g_torch = torch.sigmoid(f_torch)
    h_torch = g_torch.sum()

    # Retain gradients for all intermediate PyTorch tensors
    c_torch.retain_grad()
    d_torch.retain_grad()
    e_torch.retain_grad()
    f_torch.retain_grad()
    g_torch.retain_grad()
    h_torch.retain_grad()
    
    h_torch.backward()

    # --- Assertions ---
    # Compare forward outputs
    self.assertAlmostEqual(h_drop.item(), h_torch.item(), places=5, msg="Sum outputs do not match!")
    
    # Compare gradients for all tensors
    self.assertTrue(self._compare_grads(self.a_drop.grad, self.a_torch.grad), msg="Gradient mismatch for 'a'")
    self.assertTrue(self._compare_grads(self.b_drop.grad, self.b_torch.grad), msg="Gradient mismatch for 'b'")
    self.assertTrue(self._compare_grads(c_drop.grad, c_torch.grad), msg="Gradient mismatch for 'c'")
    self.assertTrue(self._compare_grads(d_drop.grad, d_torch.grad), msg="Gradient mismatch for 'd'")
    self.assertTrue(self._compare_grads(e_drop.grad, e_torch.grad), msg="Gradient mismatch for 'e'")
    self.assertTrue(self._compare_grads(f_drop.grad, f_torch.grad), msg="Gradient mismatch for 'f'")
    self.assertTrue(self._compare_grads(g_drop.grad, g_torch.grad), msg="Gradient mismatch for 'g'")
    self.assertTrue(self._compare_grads(h_drop.grad, h_torch.grad), msg="Gradient mismatch for 'h'")

  def _compare_grads(self, drop_grad, torch_grad, tol=1e-5):
    """
    Helper function to compare gradients between drop.tensor and PyTorch tensors.
    """
    if drop_grad is None or torch_grad is None:
      return drop_grad is None and torch_grad is None
    return np.allclose(drop_grad.tolist(), torch_grad.detach().numpy(), atol=tol)

if __name__ == '__main__':
  unittest.main()