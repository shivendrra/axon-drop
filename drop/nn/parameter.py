from ..tensor import tensor
from ..helpers.utils import randn
from ..helpers.shape import flatten

class Parameter(tensor):
  def __init__(self, shape) -> None:
    data = randn(domain=(-1, 1), shape=shape)
    super().__init__(data)
  
  def zero_grad(self) -> None:
    self.grad.zero_grad()
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return super().__repr__()
  
  def __str__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()