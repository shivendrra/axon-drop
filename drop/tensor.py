from .helpers.shape import *
from .scalar import Scalar
from .helpers.utils import zeros
class Tensor:
  def __init__(self, *data) -> None:
    data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.data = self.initialize_data(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self._backward = lambda: None
    self.grad = self._show_grad()

  def initialize_data(self, data):
    def _init(data):
      if isinstance(data, list):
        return [_init(_d) for _d in data]
      return data if isinstance(data, Scalar) else Scalar(data)
    return _init(data)
  
  def __repr__(self) -> str:
    if self.ndim == 1:
      return f"Tensor([{self.data}])"
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f"Tensor([{data_str}])"
  
  def _show_grad(self):
    def _grad(data):
      if isinstance(data, list):
        return [_grad(_d) for _d in data]
      return data.grad
    grad = _grad(self.data)
    return grad

  def shape(self):
    return get_shape(self.data)
  
  def __add__(self, other):
    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      return a + b
    return Tensor(_add(self.data, other.data))
  
  def __mul__(self, other):
    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      return a + b
    return Tensor(_mul(self.data, other.data))
  
  def __sub__(self, other):
    def _sub(a, b):
      if isinstance(a, list):
        return [_sub(_a, _b) for _a, _b in zip(a, b)]
      return a - b
    return Tensor(_sub(self.data, other.data))
  
  def __truediv__(self, other):
    def _div(a, b):
      if isinstance(a, list):
        return [_div(_a, _b) for _a, _b in zip(a, b)]
      return a + b
    return Tensor(_div(self.data, other.data))
  
  def __radd__(self, other):
    return other + self
  
  def __rmul__(self, other):
    return other * self
  
  def __rtruediv__(self, other):
    return other / self
  
  def __pow__(self, exp):
    def _pow(data):
      if isinstance(data, list):
        return [_pow(_d) for _d in data]
      return data ** exp
    return Tensor(_pow(self.data, exp))
  
  def relu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.relu()
    return Tensor(ops(self.data))
  
  def tanh(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.tanh()
    return Tensor(ops(self.data))

  # def gelu(self):
  #   def ops(data):
  #     if isinstance(data, list):
  #       return [ops(_d) for _d in data]
  #     return data.gelu()
  #   return Tensor(ops(self.data))

  # def silu(self):
  #   def ops(data):
  #     if isinstance(data, list):
  #       return [ops(_d) for _d in data]
  #     return data.silu()
  #   return Tensor(ops(self.data))

  def sigmoid(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.sigmoid()
    return Tensor(ops(self.data))

  # def swiglu(self):
  #   def ops(data):
  #     if isinstance(data, list):
  #       return [ops(_d) for _d in data]
  #     return data.swiglu()
  #   return Tensor(self.data)
  
  def backward(self):
    def _back(data):
      if isinstance(data, list):
        for _d in data:
          _back(_d)
      else:
        data.backward()
    _back(self.data)