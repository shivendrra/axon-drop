from .helpers.shape import *
from .scalar import Scalar
from .helpers.utils import zeros
from typing import *

class tensor:
  def __init__(self, *data, requires_grad=True) -> None:
    data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.data = self.initialize_data(data)
    self.shape = self.get_shape()
    self.ndim = len(self.shape)
    self.requires_grad = requires_grad
    if self.requires_grad:
      self.grad = tensor(zeros(self.shape), requires_grad=False)
    else:
      self.grad = None
    self._backward = lambda: None

  def initialize_data(self, data):
    def _init(data):
      if isinstance(data, list):
        return [_init(_d) for _d in data]
      return data if isinstance(data, Scalar) else Scalar(data)
    return _init(data)

  def __str__(self) -> str:
    def _repr(data):
      if isinstance(data, list):
        return [_repr(_d) for _d in data]
      return f"{data.data:.4f}"

    if self.ndim == 1:
      data_str = ', '.join(_repr(self.data))
      return f"tensor([{data_str}])"

    data_str = ',\n\t'.join(['[' + ', '.join(map(str, _repr(row))) + ']' for row in self.data])
    return f"tensor([{data_str}])"

  def __repr__(self) -> str:
    return f"tensor({self.data})"

  def __getitem__(self, index:tuple):
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      return data[index[-1]]
    else:
      return self.data[index]
  
  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:tuple, value: Any) -> None:
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      data[index[-1]] = value
    else:
      self.data[index] = value

  def __iter__(self) -> Iterator:
    for item in self.data:
      yield item

  def shape(self):
    return self.get_shape()
  
  def get_shape(self):
    return get_shape(self.data)

  def __add__(self, other):
    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      return a + b
    return tensor(_add(self.data, other.data))
  
  def __mul__(self, other):
    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      return a * b
    return tensor(_mul(self.data, other.data))
  
  def __sub__(self, other):
    def _sub(a, b):
      if isinstance(a, list):
        return [_sub(_a, _b) for _a, _b in zip(a, b)]
      return a - b
    return tensor(_sub(self.data, other.data))
  
  def __truediv__(self, other):
    def _div(a, b):
      if isinstance(a, list):
        return [_div(_a, _b) for _a, _b in zip(a, b)]
      return a / b
    return tensor(_div(self.data, other.data))
  
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
    return tensor(_pow(self.data, exp))
  
  def relu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.relu()
    return tensor(ops(self.data))
  
  def tanh(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.tanh()
    return tensor(ops(self.data))

  def gelu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.gelu()
    return tensor(ops(self.data))

  def silu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.silu()
    return tensor(ops(self.data))

  def sigmoid(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.sigmoid()
    return tensor(ops(self.data))

  def swiglu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.swiglu()
    return tensor(self.data)
  
  def backward(self):
    def _back(data, grad_tensor):
      if isinstance(data, list):
        for i in range(len(data)):
          print(grad_tensor)
          _back(data[i], grad_tensor[i])
      else:
        print(data)
        data.backward()
        print(data)
        grad_tensor = data.grad
        if hasattr(data, 'prev'):
          for prev in data.prev:
            prev.backward()

    grad_tensor = tensor(zeros(self.shape), requires_grad=False).data
    _back(self.data, grad_tensor)

    self.grad = tensor(grad_tensor, requires_grad=False)