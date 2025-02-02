from .._tensor import tensor
from .module import Module

class ReLU(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace

  def forward(self, x):
    if self.inplace:
      x = x.relu()
      return x
    else:
      out = x.relu()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<Relu(inplace={self.inplace})>"

class Tanh(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace

  def forward(self, x):
    if self.inplace:
      x = x.tanh()
      return x
    else:
      out = x.tanh()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<Tanh(inplace={self.inplace})>"

class GELU(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace

  def forward(self, x):
    if self.inplace:
      x = x.gelu()
      return x
    else:
      out = x.gelu()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<Gelu(inplace={self.inplace})>"

class Silu(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace
    
  def forward(self, x):
    if self.inplace:
      x = x.silu()
      return x
    else:
      out = x.silu()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<SiLU(inplace={self.inplace})>"

class Sigmoid(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace
    
  def forward(self, x):
    if self.inplace:
      x = x.sigmoid()
      return x
    else:
      out = x.sigmoid()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<Sigmoid(inplace={self.inplace})>"

class SwiGLU(Module):
  def __init__(self, inplace=None) -> None:
    super().__init__()
    self.inplace = inplace
    
  def forward(self, x):
    if self.inplace:
      x = x.swiglu()
      return x
    else:
      out = x.swiglu()
    return out
  
  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    return self.forward(x)
  
  def __repr__(self):
    return f"<SwiGLU(inplace={self.inplace})>"