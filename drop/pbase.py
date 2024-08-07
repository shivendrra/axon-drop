import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'libvalue.dll')
lib = ctypes.CDLL(lib_path)

class DType:
  INT8 = 0
  INT16 = 1
  INT32 = 2
  INT64 = 3
  FLOAT16 = 4
  FLOAT32 = 5
  FLOAT64 = 6

class Value(ctypes.Structure):
  pass

Value._fields_ = [
  ("data", ctypes.c_void_p),
  ("grad", ctypes.c_void_p),
  ("_prev", ctypes.POINTER(ctypes.POINTER(Value))),
  ("_prev_size", ctypes.POINTER(ctypes.c_int)),
  ("_backward", ctypes.CFUNCTYPE(None, ctypes.POINTER(Value))),
  ("exp", ctypes.POINTER(ctypes.c_double)),
  ("dtype", ctypes.c_int),
]

lib.initialize_value.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.initialize_value.restype = ctypes.POINTER(Value)

lib.get_data_as_double.argtypes = [ctypes.POINTER(Value)]
lib.get_data_as_double.restype = ctypes.c_double

lib.get_grad_as_double.argtypes = [ctypes.POINTER(Value)]
lib.get_grad_as_double.restype = ctypes.c_double

lib.add_val.argtypes = [ctypes.POINTER(Value), ctypes.POINTER(Value)]
lib.add_val.restype = ctypes.POINTER(Value)

lib.mul_val.argtypes = [ctypes.POINTER(Value), ctypes.POINTER(Value)]
lib.mul_val.restype = ctypes.POINTER(Value)

lib.pow_val.argtypes = [ctypes.POINTER(Value), ctypes.POINTER(ctypes.c_double)]
lib.pow_val.restype = ctypes.POINTER(Value)

lib.relu.argtypes = [ctypes.POINTER(Value)]
lib.relu.restype = ctypes.POINTER(Value)

lib.backward.argtypes = [ctypes.POINTER(Value)]
lib.backward.restype = None

class ValueWrapper:
  def __init__(self, data, dtype):
    self.dtype = dtype
    if dtype == DType.FLOAT64:
      c_data = ctypes.c_double(data)
    elif dtype == DType.FLOAT32:
      c_data = ctypes.c_float(data)
    elif dtype == DType.INT64:
      c_data = ctypes.c_int64(data)
    elif dtype == DType.INT32:
      c_data = ctypes.c_int32(data)
    elif dtype == DType.INT16:
      c_data = ctypes.c_int16(data)
    elif dtype == DType.INT8:
      c_data = ctypes.c_int8(data)
    else:
      raise ValueError(f"Unsupported dtype: {dtype}")
      
    # Debug: Print the data being passed
    print(f"Data: {c_data.value}, DType: {self.dtype}")
    
    self.value = lib.initialize_value(ctypes.byref(c_data), dtype)
    
    # Debug: Print the result from the library call
    print(f"Library call result: {self.value}")

  def data(self):
    return lib.get_data_as_double(self.value)

  def grad(self):
    return lib.get_grad_as_double(self.value)

  def add(self, other):
    return ValueWrapper.from_pointer(lib.add_val(self.value, other.value), self.dtype)

  def mul(self, other):
    return ValueWrapper.from_pointer(lib.mul_val(self.value, other.value), self.dtype)

  def pow(self, exp):
    exp = ctypes.c_double(exp)
    return ValueWrapper.from_pointer(lib.pow_val(self.value, ctypes.byref(exp)), self.dtype)

  def relu(self):
    return ValueWrapper.from_pointer(lib.relu(self.value), self.dtype)

  def backward(self):
    lib.backward(self.value)

  @classmethod
  def from_pointer(cls, ptr, dtype):
    obj = cls.__new__(cls)
    obj.value = ptr
    obj.dtype = dtype
    return obj

if __name__ == "__main__":
  val1 = ValueWrapper(2.0, DType.FLOAT64)
  val2 = ValueWrapper(3.0, DType.FLOAT64)

  result = val1.add(val2)
  print(f"Addition result: {result.data()}")  # Should output 5.0

  result = val1.mul(val2)
  print(f"Multiplication result: {result.data()}")  # Should output 6.0

  result = val1.pow(3.0)
  print(f"Power result: {result.data()}")  # Should output 8.0

  val3 = ValueWrapper(-1.0, DType.FLOAT64)
  result = val3.relu()
  print(f"ReLU result: {result.data()}")  # Should output 0.0

  val4 = ValueWrapper(1.0, DType.FLOAT64)
  val5 = ValueWrapper(2.0, DType.FLOAT64)
  result = val4.add(val5)
  result.backward()
  print(f"Gradient of val4 after backward: {val4.grad()}")  # Should output 1.0
  print(f"Gradient of val5 after backward: {val5.grad()}")  # Should output 1.0