from .helpers.shape import *
from .scalar import scalar
from .helpers.utils import zeros
from typing import *
from .helpers.ops import *

def compute_grad(self):
  def _compute_grad(data):
    if isinstance(data, list):
      return [_compute_grad(_d) for _d in data]
    return data.grad
  return tensor(_compute_grad(self.data), requires_grad=False)

def initialize_data(data, dtype):
  def _init(data):
    if isinstance(data, list):
      return [_init(_d) for _d in data]
    return data if isinstance(data, scalar) else scalar(data, dtype)
  return _init(data)

class tensor:
  def __init__(self, *data, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None, requires_grad=True) -> None:
    data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.data = initialize_data(data, dtype)
    self.dtype = dtype
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.requires_grad = requires_grad
    self.prev = set() if requires_grad else None
    self.grad_fn = "<NotSet>"

  @property
  def grad(self):
    return compute_grad(self)

  def __str__(self) -> str:
    def _repr(data):
      if isinstance(data, list):
        return [_repr(_d) for _d in data]
      return f"{data.data:.4f}"

    if self.ndim == 1:
      data_str = ', '.join(_repr(self.data))
      return f"tensor([{data_str}])"

    data_str = ',\n\t'.join(['[' + ', '.join(map(str, _repr(row))) + ']' for row in self.data])
    return f"tensor([{data_str}])" if self.grad_fn == "<NotSet>" else f"tensor([{data_str}], grad_fn={self.grad_fn})"

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
    return get_shape(self.data)

  def __add__(self, other):
    def _add(a, b):
      if isinstance(a, list):
        return [_add(_a, _b) for _a, _b in zip(a, b)]
      return a + b
    out = tensor(_add(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<AddBackward>"
    return out
  
  def __mul__(self, other):
    def _mul(a, b):
      if isinstance(a, list):
        return [_mul(_a, _b) for _a, _b in zip(a, b)]
      return a * b
    out = tensor(_mul(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<MulBackward>"
    return out
  
  def __matmul__(self, other):
    out = tensor(matmul(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<MatMulBackward>"
    return out

  def __sub__(self, other):
    def _sub(a, b):
      if isinstance(a, list):
        return [_sub(_a, _b) for _a, _b in zip(a, b)]
      return a - b
    out = tensor(_sub(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<SubBackward>"
    return out
  
  def __truediv__(self, other):
    def _div(a, b):
      if isinstance(a, list):
        return [_div(_a, _b) for _a, _b in zip(a, b)]
      return a / b
    out = tensor(_div(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<DivBackward>"
    return out
  
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
    out = tensor(_pow(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<PowBackward>"
    return out
  
  def relu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.relu()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<ReluBackward>"
    return out
  
  def tanh(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.tanh()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<TanhBackward>"
    return out

  def gelu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.gelu()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<geluBackward>"
    return out

  def silu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.silu()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SiluBackward>"
    return out

  def sigmoid(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.sigmoid()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SigmoidBackward>"
    return out

  def swiglu(self):
    def ops(data):
      if isinstance(data, list):
        return [ops(_d) for _d in data]
      return data.swiglu()
    out = tensor(ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SwigluBackward>"
    return out
  
  def sum(self, axis=None, keepdims=False):
    if axis == None:
      if keepdims:
        out = [[sum(flatten(self.data))]]
      else:
        out = sum(flatten(self.data))
    elif axis == 0:
      out = sum_axis0(self.data)
    else:
      out = sum_axis(self.data, axis, keepdims)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SumBackward>"
    return out

  def backward(self):
    if self.grad_fn != "<SumBackward>":
      raise ValueError("Backward can only be called through 'Sum' function")
    else:
      if self.requires_grad:
        self.data[0].backward()
      else:
        raise ValueError("requires_grad is set to False, grads can't be computed")