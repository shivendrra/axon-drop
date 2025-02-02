import ctypes
from ._core import CScalar, CTensor, libtensor
from ._core import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
from typing import *
from ._helpers import flatten, get_shape

int8, int16, int32, int64, float32, float64 = DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_FLOAT32, DTYPE_FLOAT64

class tensor:
  int8, int16, int32, int64, float32, float64 = int8, int16, int32, int64, float32, float64
  def __init__(self, data=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None, requires_grad:bool=False):
    if data is not None:
      data, shape, dtype = flatten(data), get_shape(data), dtype if dtype else float32
      
      self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
      self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
      self._dtype_ctype = ctypes.c_int(dtype)
      self._ndim_ctype = ctypes.c_int(len(shape))

      self.shape, self.ndim = shape.copy(), len(shape)
      self.numel = 1
      self.dtype = dtype
      for i in self.shape:
        self.numel *= i

      self.requires_grads = requires_grad
      self.tensor = libtensor.create_tensor(self._data_ctype, self._shape_ctype, self._ndim_ctype, self._dtype_ctype)
    else:
      self.tensor, self.shape, self.ndim, self.requires_grads, self.dtype = None, None, None, None, None

  def __del__(self):
    libtensor.delete_strides(self.tensor)
    libtensor.delete_backstrides(self.tensor)
    libtensor.delete_shape(self.tensor)
    libtensor.delete_data(self.tensor)
    libtensor.delete_tensor(self.tensor)
  
  def __str__(self):
    libtensor.print_tensor(self.tensor)

  def __add__(self, other):
    pass