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
    return self.value.contents.data

  @data.setter
  def data(self, new_data):
    self.value.contents.data = new_data

  @property
  def grad(self):
    return self.value.contents.grad
  
  @grad.setter
  def grad(self, new_grad):
    self.value.contents.grad = new_grad

  def __repr__(self):
    return f"Scalar(data={self.data:.4f}, grad={self.grad:.4f})"

  def __str__(self):
    return self.__repr__()

  def __add__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = lib.initialize_scalars(float(other), None, 0)
    out = lib.add_val(self.value, other.value)
    return Scalar(out.contents.data)
    
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = lib.initialize_scalars(float(other), None, 0)
    out = lib.mul_val(self.value, other.value)
    return Scalar(out.contents.data)
    
  def __rmul__(self, other):
    return self * other

  def __pow__(self, exp):
    out = lib.pow_val(self.value, ctypes.c_float(exp))
    return Scalar(out.contents.data)

  def __neg__(self):
    out = lib.negate(self.value)
    return Scalar(out.contents.data)

  def __sub__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = lib.initialize_scalars(float(other), None, 0)
    out = lib.sub_val(self.value, other.value)
    return Scalar(out.contents.data)
    
  def __rsub__(self, other):
    return other - self

  def __truediv__(self, other):
    if isinstance(other, Scalar):
      other = other
    else:
      other = lib.initialize_scalars(float(other), None, 0)
    out = lib.div_val(self.value, other.value)
    return Scalar(out.contents.data)

  def __rtruediv__(self, other):
    return other / self

  def relu(self):
    out = lib.relu(self.value)
    return Scalar(out.contents.data)

  def sigmoid(self):
    out = lib.sigmoid(self.value)
    return Scalar(out.contents.data)
  
  def tanh(self):
    out = lib.tanh(self.value)
    return Scalar(out.contents.data)
  
  def backward(self):
    lib.backward(self.value)

