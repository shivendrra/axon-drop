from ._core import CScalar, libscalar
from ._core import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
import ctypes
from typing import *

int8 = DTYPE_INT8
int16 = DTYPE_INT16
int32 = DTYPE_INT32
int64 = DTYPE_INT64
float32 = DTYPE_FLOAT32
float64 = DTYPE_FLOAT64

class scalar:
  int8 = int8
  int16 = int16 
  int32 = int32 
  int64 = int64 
  float32 = float32
  float64 = float64

  def __init__(self, data:Union[int, float], dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None):
    if isinstance(data, CScalar):
      self.value = data
    else:
      dtype = dtype if dtype is not None else DTYPE_FLOAT32
      self.value = libscalar.initialize_scalars(ctypes.c_float(data), ctypes.c_int(dtype), None, 0)
    self.prev = set()

  @property
  def data(self):
    return libscalar.get_scalar_data(self.value)

  @data.setter
  def data(self, new_data):
    libscalar.set_scalar_data(self.value, new_data)

  @property
  def grad(self):
    return libscalar.get_scalar_grad(self.value)
  
  @grad.setter
  def grad(self, new_grad):
    libscalar.set_scalar_grad(self.value, new_grad)

  @property
  def dtype(self):
    if isinstance(self.value, CScalar):
      dtype = self.value.dtype
    else:
      dtype = self.value.contents.dtype
    if dtype == 0:
      return f"int8"
    elif dtype == 1:
      return f"int16"
    elif dtype == 2:
      return f"int32"
    elif dtype == 3:
      return f"int64"
    elif dtype == 4:
      return f"float32"
    elif dtype == 5:
      return f"float64"

  def zero_grad(self):
    self.grad = 0.0

  def __repr__(self):
    return f"Scalar(data={self.data}, grad={self.grad})"

  def __str__(self):
    data_value = libscalar.get_scalar_data(self.value)
    grad_value = libscalar.get_scalar_grad(self.value)
    return f"Scalar(data={data_value:.4f}, grad={grad_value:.4f}, dtype={self.dtype})"

  def __add__(self, other):
    if isinstance(other, scalar):
      other = other
    else:
      other = scalar(other)
    out = scalar(libscalar.add_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, scalar):
      other = other
    else:
      other = scalar(other)
    out = scalar(libscalar.mul_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __rmul__(self, other):
    return self * other

  def __pow__(self, exp):
    out = scalar(libscalar.pow_val(self.value, exp).contents)
    out.prev = (self)
    return out

  def __neg__(self):
    out = scalar(libscalar.negate(self.value).contents)
    out.prev = (self, )
    return out

  def __sub__(self, other):
    if isinstance(other, scalar):
      other = other
    else:
      other = scalar(other)
    out = scalar(libscalar.sub_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out
    
  def __rsub__(self, other):
    return - (self - other)

  def __truediv__(self, other):
    if isinstance(other, scalar):
      other = other
    else:
      other = scalar(other)
    out = scalar(libscalar.div_val(self.value, other.value).contents)
    out.prev = (self, other)
    return out

  def __rtruediv__(self, other):
    return other / self
  
  def log(self):
    out = scalar(libscalar.log_val(self.value).contents)
    out.prev = (self, )
    return out

  def relu(self):
    out = scalar(libscalar.relu(self.value).contents)
    out.prev = (self, )
    return out

  def sigmoid(self):
    out = scalar(libscalar.sigmoid(self.value).contents)
    out.prev = (self, )
    return out
  
  def tanh(self):
    out = scalar(libscalar.tan_h(self.value).contents)
    out.prev = (self, )
    return out
  
  def gelu(self):
    out = scalar(libscalar.gelu(self.value).contents)
    out.prev = (self, )
    return out
  
  def silu(self):
    out = scalar(libscalar.silu(self.value).contents)
    out.prev = (self, )
    return out

  def swiglu(self):
    out = scalar(libscalar.swiglu(self.value).contents)
    out.prev = (self, )
    return out
  
  def backward(self):
    libscalar.backward(self.value)