import ctypes
from _core import CScalar, CTensor, libtensor
from _core import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
from typing import *
from _helpers import flatten, get_shape, broadcast_shape
import os, sys, io

int8, int16, int32, int64, float32, float64 = DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_FLOAT32, DTYPE_FLOAT64

class tensor:
  int8, int16, int32, int64, float32, float64 = int8, int16, int32, int64, float32, float64
  def __init__(self, data=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None, requires_grad:bool=False):
    if data is not None:
      if isinstance(data, CTensor):
        self.tensor = data
        self.ndim, self.dtype, self.numel = self.tensor.ndim, self.tensor.dtype, self.tensor.size

      data, shape, dtype = flatten(data), get_shape(data), dtype if dtype else float32
      self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
      self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
      self._dtype_ctype = ctypes.c_int(dtype)
      self._ndim_ctype = ctypes.c_int(len(shape))
      self.shape, self.ndim = shape.copy(), len(shape)
      self.numel = 1
      self.dtype = dtype
      for i in self.shape:
        self.numel *= i

      self.requires_grads = requires_grad
      self.tensor = libtensor.create_tensor(self._data_ctype, self._shape_ctype, self._ndim_ctype, self._dtype_ctype)
    else:
      self.tensor, self.shape, self.ndim, self.requires_grads, self.dtype = None, None, None, None, None

  @classmethod
  def init_from_c_tensor(cls, c_tensor_ptr, dtype, requires_grad):
    if not c_tensor_ptr:
      raise ValueError("Received NULL pointer from C tensor operation.")

    c_tensor = c_tensor_ptr.contents  # dereferencing the pointer
    tensor_instance = cls()
    tensor_instance.tensor = c_tensor_ptr   # original pointer stored
    tensor_instance.shape = [c_tensor.shape[i] for i in range(c_tensor.ndim)]
    tensor_instance.ndim = c_tensor.ndim
    tensor_instance.dtype = dtype
    tensor_instance.numel = c_tensor.size
    tensor_instance.requires_grads = requires_grad
    return tensor_instance

  def __del__(self):
    if self.tensor:
      libtensor.delete_tensor(self.tensor)

  def __str__(self):
    if not self.tensor:
      return "tensor(None)"

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    libtensor.print_tensor(self.tensor)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return output.strip()
  
  def transpose(self):
    out = libtensor.transpose_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def flatten(self):
    out = libtensor.flatten_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other, self.dtype, self.requires_grads)
    if self.shape != other.shape:
      _, requires_broadcasting = broadcast_shape(self.shape, other.shape)
      if requires_broadcasting:
        out = libtensor.add_broadcasted_tensor(self.tensor, other.tensor)
      else:
        raise ValueError(f"Shapes don't match {self.shape} != {other.shape}, hence can't perform the operation!")
    else:
      out = libtensor.add_tensor(self.tensor, other.tensor)
    if not out:
      raise RuntimeError("Failed to create new tensor in C backend!")
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def __sub__(self, other):
    other = other if isinstance(other, tensor) else tensor(other, self.dtype, self.requires_grads)
    if self.shape != other.shape:
      _, requires_broadcasting = broadcast_shape(self.shape, other.shape)
      if requires_broadcasting:
        out = libtensor.sub_broadcasted_tensor(self.tensor, other.tensor)
      else:
        raise ValueError(f"Shapes don't match {self.shape} != {other.shape}, hence can't perform the operation!")
    else:
      out = libtensor.sub_tensor(self.tensor, other.tensor)
    if not out:
      raise RuntimeError("Failed to create new tensor in C backend!")
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def __mul__(self, other):
    other = other if isinstance(other, tensor) else tensor(other, self.dtype, self.requires_grads)
    if self.shape != other.shape:
      _, requires_broadcasting = broadcast_shape(self.shape, other.shape)
      if requires_broadcasting:
        out = libtensor.elemwise_mul_broadcasted_tensor(self.tensor, other.tensor)
      else:
        raise ValueError(f"Shapes don't match {self.shape} != {other.shape}, hence can't perform the operation!")
    else:
      out = libtensor.elemwise_mul_tensor(self.tensor, other.tensor)
    if not out:
      raise RuntimeError("Failed to create new tensor in C backend!")
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def __truediv__(self, other):
    other = other if isinstance(other, tensor) else tensor(other, self.dtype, self.requires_grads)
    if self.shape != other.shape:
      _, requires_broadcasting = broadcast_shape(self.shape, other.shape)
      if requires_broadcasting:
        out = libtensor.div_broadcasted_tensor(self.tensor, other.tensor)
      else:
        raise ValueError(f"Shapes don't match {self.shape} != {other.shape}, hence can't perform the operation!")
    else:
      out = libtensor.div_tensor(self.tensor, other.tensor)
    if not out:
      raise RuntimeError("Failed to create new tensor in C backend!")
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other
  
  def __rsub__(self, other):
    return other - self
  
  def __rtruediv__(self, other):
    return other / self

  def __pow__(self, exp):
    pass

  def relu(self):
    out = libtensor.relu_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def tanh(self):
    out = libtensor.tanh_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def cos(self):
    out = libtensor.cos_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def sin(self):
    out = libtensor.sin_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def silu(self):
    out = libtensor.silu_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def sigmoid(self):
    out = libtensor.sigmoid_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def gelu(self):
    out = libtensor.gelu_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def swiglu(self):
    out = libtensor.swiglu_tensor(self.tensor)
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def sum(self, axis:int=-1, keepdims:bool=False):
    # axis = axis if axis > 0 else self.ndim + axis
    out = libtensor.sum_tensor(self.tensor, ctypes.c_int(axis), ctypes.c_bool(keepdims))
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def max(self, axis:int=-1, keepdims:bool=False):
    # axis = axis if axis > 0 else self.ndim + axis
    out = libtensor.max_tensor(self.tensor, ctypes.c_int(axis), ctypes.c_bool(keepdims))
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def min(self, axis:int=-1, keepdims:bool=False):
    # axis = axis if axis > 0 else self.ndim + axis
    out = libtensor.min_tensor(self.tensor, ctypes.c_int(axis), ctypes.c_bool(keepdims))
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)
  
  def reshape(self, shape:Union[tuple, list]):
    new_shape = (ctypes.c_int * len(shape))(*list(shape).copy())
    new_ndim = len(shape)
    out = libtensor.reshape_tensor(self.tensor, new_shape, ctypes.c_int(new_ndim))
    return tensor.init_from_c_tensor(out, self.dtype, self.requires_grads)

  def backward(self):
    libtensor.tensor_backward(self.tensor)

if __name__ == "__main__":
  a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]], float32, True)
  b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]], float32, True)
  print(a)
  print(b)
  result = (a + b)
  c = result.sum()
  print(result)
  print(c)