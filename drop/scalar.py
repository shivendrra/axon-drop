from .cbase import CScalar, lib
import ctypes

class Scalar:
  def __init__(self, data):
    if isinstance(data, CScalar):
      self.value = data
    else:
      self.value = lib.initialize_scalars(float(data), None, 0)

  @property
  def data(self):
    if isinstance(self.value, CScalar):
      return self.value.data
    else:
      return self.value.contents.data

  @data.setter
  def data(self, new_data):
    if isinstance(self.value, CScalar):
      self.value.data = float(new_data)
    else:
      self.value.contents.data = float(new_data)

  @property
  def grad(self):
    if isinstance(self.value, CScalar):
      return self.value.grad
    else:
      return self.value.contents.grad
  
  @grad.setter
  def grad(self, new_grad):
    if isinstance(self.value, CScalar):
      self.value.grad = float(new_grad)
    else:
      self.value.contents.grad = float(new_grad)

  def __repr__(self):
    return f"Scalar(data={self.data:.4f}, grad={self.grad:.4f})"

  def __str__(self):
    return self.__repr__()

  def __add__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = lib.add_val(self.value, other.value)
    return Scalar(out.contents)
    
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = lib.mul_val(self.value, other.value)
    return Scalar(out.contents)
    
  def __rmul__(self, other):
    return self * other

  def __pow__(self, exp):
    out = lib.pow_val(self.value, ctypes.c_float(exp))
    return Scalar(out.contents)

  def __neg__(self):
    out = lib.negate(self.value)
    return Scalar(out.contents)

  def __sub__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = lib.sub_val(self.value, other.value)
    return Scalar(out.contents)
    
  def __rsub__(self, other):
    return - (self - other)

  def __truediv__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = Scalar(other)
    out = lib.div_val(self.value, other.value)
    return Scalar(out.contents)

  def __rtruediv__(self, other):
    return other / self

  def relu(self):
    out = lib.relu(self.value)
    return Scalar(out.contents)

  def sigmoid(self):
    out = lib.sigmoid(self.value)
    return Scalar(out.contents)
  
  def tanh(self):
    out = lib.tan_h(self.value)
    return Scalar(out.contents)
  
  def gelu(self):
    out = lib.gelu(self.value)
    return Scalar(out.contents)
  
  def silu(self):
    out = lib.silu(self.value)
    return Scalar(out.contents)

  def swiglu(self):
    out = lib.swiglu(self.value)
    return Scalar(out.contents)
  
  def backward(self):
    lib.backward(self.value)