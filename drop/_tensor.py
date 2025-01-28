import ctypes
from ._core import CScalar, CTensor, libtensor
from ._core import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
from typing import *
from .helpers.shape import flatten, get_shape

int8, int16, int32, int64, float32, float64 = DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_FLOAT32, DTYPE_FLOAT64

class tensor:
  int8, int16, int32, int64, float32, float64 = int8, int16, int32, int64, float32, float64
  def __init__(self, data=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None, device:Optional[Literal['cuda', 'cpu']]='cpu', requires_grad:bool=False):
    if data is not None:
      data, shape, dtype = flatten(data), get_shape(data), dtype if dtype else float32
      
      self._data_ctype = (ctypes.c_float * len(data))(*data.copy())
      self._shape_ctype = (ctypes.c_int * len(shape))(*shape.copy())
      self._dtype_ctype = ctypes.c_int(dtype)
      self._ndim_ctype = ctypes.c_int(len(shape))
      self._device_ctype = device.encode('utf-8')

      self.shape, self.ndim, self.device = shape.copy(), len(shape), device
      self.numel = 1
      self.dtype = dtype
      for i in self.shape:
        self.numel *= i
      
      self.requires_grads = requires_grad
      self.hooks = []
      self.grad, self.grad_fn = None, None
      self.tensor = libtensor.create_tensor(self._data_ctype, self._shape_ctype, self._ndim_ctype, self._device_ctype, self._dtype_ctype)
    else:
      self.tensor, self.shape, self.ndim, self.device, self.requires_grads, self.dtype = None, None, None, device, None, None
      self.hooks, self.grad, self.grad_fn = [], None, None

  def __del__(self):
    libtensor.delete_strides(self.tensor)
    libtensor.delete_backstrides(self.tensor)
    libtensor.delete_shape(self.tensor)
    libtensor.delete_device(self.tensor)
    libtensor.delete_aux(self.tensor)
    libtensor.delete_data(self.tensor)
    libtensor.delete_tensor(self.tensor)
  
  def to(self, device:str):
    device = str(device)
    self.device = device
    self._device_ctype = self.device.encode('utf-8')
    libtensor.to_device(self.tensor, self._device_ctype)
    return self