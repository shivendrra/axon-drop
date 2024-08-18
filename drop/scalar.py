from .cbase import CScalar, lib

class Scalar:
  def __init__(self, data):
    if isinstance(data, CScalar):
      self.value = data
    else:
      self.value = lib.initialize_scalars(float(data), None, 0)
    self.prev = set()

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