from ._scalar import scalar, int8, int16, int32, int64, float32, float64
from ._ops import *
from ._tensor import tensor
from ._random import RNG, random
from ._utils import (
  _zeros as zeros,
  _ones as ones,
  _randint as randint,
  _arange as arange,
  _randn as randn,
  _ones_like as ones_like,
  _zeros_like as zeros_like
)

__all__ = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']