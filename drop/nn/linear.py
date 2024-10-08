from ..tensor import tensor
from .parameter import Parameter
from .module import Module
from ..helpers.utils import _randn

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    super(Linear, self).__init__()
    self.wei = Parameter(_randn(shape=(_in, _out)))
    if bias:
      self.bias = Parameter(_randn(shape=(1, _out)))
    else:
      self.bias = None

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    out = x @ self.wei.data
    if self.bias is not None:
      out = out + self.bias.data
    return out

  def parameters(self):
    params = [self.wei]
    if self.bias is not None:
      params.append(self.bias)
    return params
  
  def __repr__(self):
    return f"<LinearLayer in_features={self.wei.shape[0]} out_features={self.wei.shape[1]}>"