import engine

class value:
  def __init__(self, data):
    if isinstance(data, engine.Value):
      self.value = data
    else:
      self.value = engine.Value(float(data))
  
  @property
  def data(self):
    return self.value.data

  @data.setter
  def data(self, new_data):
    self.value.data = new_data

  @property
  def grad(self):
    return self.value.grad
  
  @grad.setter
  def grad(self, new_data):
    self.value.grad = new_data

  def __add__(self, other):
    if isinstance(other, value):
      return value(engine.Value.add(self.value, other.value))
    return value(engine.Value.add(self.value, engine.Value(float(other))))
    
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, value):
      return value(engine.Value.mul(self.value, other.value))
    return value(engine.Value.mul(self.value, engine.Value(float(other))))
    
  def __rmul__(self, other):
    return self * other

  def __pow__(self, exp):
    return value(engine.Value.pow(self.value, exp))

  def __neg__(self):
    return value(engine.Value.negate(self.value))

  def __sub__(self, other):
    if isinstance(other, value):
      return value(engine.Value.sub(self.value, other.value))
    return value(engine.Value.sub(self.value, engine.Value(float(other))))
    
  def __rsub__(self, other):
    return value(engine.Value.sub(engine.Value(float(other)), self.value))

  def __truediv__(self, other):
    if isinstance(other, value):
      return value(engine.Value.truediv(self.value, other.value))
    return value(engine.Value.truediv(self.value, engine.Value(float(other))))

  def __rtruediv__(self, other):
    return value(engine.Value.truediv(engine.Value(float(other)), self.value))

  def relu(self):
    return value(engine.Value.relu(self.value))

  def sigmoid(self):
    return value(engine.Value.sigmoid(self.value))
  
  def tanh(self):
    return value(engine.Value.tanh(self.value))
  
  def gelu(self):
    return value(engine.Value.gelu(self.value))
  
  def silu(self):
    return value(engine.Value.silu(self.value))
  
  def swiglu(self):
    return value(engine.Value.swiglu(self.value))

  def backward(self):
    engine.Value.backward(self.value)

  def __repr__(self):
    return f"Value(data={self.value.data}, grad={self.value.grad})"

  def __str__(self):
    return self.__repr__()