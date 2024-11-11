from base import CTensor, lib
from base import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
import ctypes, io, sys
from typing import *

int8 = DTYPE_INT8
int16 = DTYPE_INT16
int32 = DTYPE_INT32
int64 = DTYPE_INT64
float32 = DTYPE_FLOAT32
float64 = DTYPE_FLOAT64

def _init_tensor(data:Union[float, int]) -> ctypes.c_double:
  if isinstance(data, list):
    return [_init_tensor(d) for d in data]
  return ctypes.c_double(data)

def _get_shape(data:Union[list, int]) -> list:
  if isinstance(data, list):
    return [len(data), ] + _get_shape(data[0])
  return []

def _flatten(data:list) -> list:
  if isinstance(data, list):
    return [item for sublist in data for item in _flatten(sublist)]
  return [data]

class tensor:
  int8 = int8
  int16 = int16 
  int32 = int32 
  int64 = int64 
  float32 = float32
  float64 = float64

  def __init__(self, data:Union[int, float], dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float32', 'float64']]=None):
    if isinstance(data, CTensor):
      self.data = data
    else:
      dtype = dtype if dtype is not None else DTYPE_FLOAT32
      shape = _get_shape(data)
      shape_ctypes = (ctypes.c_int * len(shape))(*shape)  # Convert shape list to a ctypes array
      flat_data = _flatten(_init_tensor(data))  # Flatten and convert the data to ctypes
      data_ctypes = (ctypes.c_double * len(flat_data))(*flat_data)

      self.data = lib.initialize_tensor(
        data_ctypes,
        ctypes.c_int(dtype),
        shape_ctypes,
        ctypes.c_int(len(shape))
      )
    self.shape = shape
    self.ndim, self.size = len(self.shape), tuple(self.shape)
    self.prev, self.grad_fn = None, "<NotSet>"
    self.strides = lib.calculate_strides(self.data)
  
  def __str__(self) -> str:
    # Create a buffer to capture the output
    buffer = io.StringIO()
    # Redirect stdout to the buffer
    sys.stdout = buffer
    try:
      lib.print_tensor(self.data)  # Call the function that prints the tensor
      output = buffer.getvalue()  # Get the output from the buffer
    finally:
      sys.stdout = sys.__stdout__  # Reset stdout
    return output.strip()  # Return the captured output as a string

  def __repr__(self) -> str:
    return f"tensor({self.data})"

  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(lib.add_tensor(self.data, other.data))
    out.prev, out.grad_fn = (self, other), "<AddBackwards>"
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(lib.mul_tensor(self.data, other.data))
    out.prev, out.grad_fn = (self, other), "<MulBackwards>"
    return out
  
  def __radd__(self, other):
    return other + self
  
  def __rmul__(self, other):
    return other * self


x = [[1, 3, 5, 5], [1, 3, 5, 5]]
y = [4, 0, 2, -5]

x, y = tensor(x, tensor.int8), tensor(y, tensor.int8)
print(x, '\n', y)