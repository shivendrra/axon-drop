from .cbase import CScalar, lib
from .cbase import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
import ctypes
from typing import *

int8 = DTYPE_INT8
int16 = DTYPE_INT16
int32 = DTYPE_INT32
int64 = DTYPE_INT64
float32 = DTYPE_FLOAT32
float64 = DTYPE_FLOAT64

class Scalar:
  int8 = int8
  int16 = int16 
  int32 = int32 
  int64 = int64 
  float32 = float32
  float64 = float64

  def __init__(self, data:Union[int, float], dtype=None):
    if isinstance(data, CScalar):
      self.value = data
    else:
      dtype = dtype if dtype is not None else DTYPE_FLOAT32
      self.value = lib.initialize_scalars(ctypes.c_double(data), ctypes.c_int(dtype), None, 0)
    self.prev = set()

  @property
  def data(self):
    return lib.get_scalar_data(self.value)

  @data.setter
  def data(self, new_data):
    lib.set_scalar_data(self.value, new_data)

  @property
  def grad(self):
    return lib.get_scalar_grad(self.value)
  
  @grad.setter
  def grad(self, new_grad):
    lib.set_scalar_grad(self.value, new_grad)

  @property
  def dtype(self):
    if isinstance(self.value, CScalar):
      dtype = self.value.dtype
    else:
      dtype = self.value.contents.dtype
    if dtype == 0:
      return f"scalar.int8"
    elif dtype == 1:
      return f"scalar.int16"
    elif dtype == 2:
      return f"scalar.int32"
    elif dtype == 3:
      return f"scalar.int64"
    elif dtype == 4:
      return f"scalar.float32"
    elif dtype == 5:
      return f"scalar.float64"

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    data_value = lib.get_scalar_data(self.value)
    grad_value = lib.get_scalar_grad(self.value)
    return f"Scalar(data={data_value:.4f}, grad={grad_value:.4f})"

  def __add__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = Scalar(lib.add_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = Scalar(lib.mul_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __rmul__(self, other):
    return self * other

  def __pow__(self, exp):
    out = Scalar(lib.pow_val(self.value, exp).contents)
    out.prev = (self)
    return out

  def __neg__(self):
    out = Scalar(lib.negate(self.value).contents)
    out.prev = (self, )
    return out

  def __sub__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = Scalar(lib.sub_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __rsub__(self, other):
    return - (self - other)

  def __truediv__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = Scalar(lib.div_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out

  def __rtruediv__(self, other):
    return other / self

  def relu(self):
    out = Scalar(lib.relu(self.value).contents)
    out.prev = (self, )
    return out

  def sigmoid(self):
    out = Scalar(lib.sigmoid(self.value).contents)
    out.prev = (self, )
    return out
  
  def tanh(self):
    out = Scalar(lib.tan_h(self.value).contents)
    out.prev = (self, )
    return out
  
  def gelu(self):
    out = Scalar(lib.gelu(self.value).contents)
    out.prev = (self, )
    return out
  
  def silu(self):
    out = Scalar(lib.silu(self.value).contents)
    out.prev = (self, )
    return out

  def swiglu(self):
    out = Scalar(lib.swiglu(self.value).contents)
    out.prev = (self, )
    return out
  
  def backward(self):
    lib.backward(self.value)