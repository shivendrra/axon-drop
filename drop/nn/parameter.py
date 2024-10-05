from ..tensor import tensor
from ..helpers.shape import flatten

class Parameter(tensor):
  def __init__(self, data) -> None:
    super().__init__(data, requires_grad=True)
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()
  
  def __str__(self) -> str:
    return "\nParameters :\n" + super().__str__()