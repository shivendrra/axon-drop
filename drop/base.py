import ctypes, os

DTYPE_INT8 = 0
DTYPE_INT16 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3
DTYPE_FLOAT32 = 4
DTYPE_FLOAT64 = 5

lib_path = os.path.join(os.path.dirname(__file__), './build/libtensor.so')
lib = ctypes.CDLL(lib_path)

class CTensor(ctypes.Structure):
  pass

CTensor._fields_ = [
  ('data', ctypes.POINTER(ctypes.c_void_p)),   # Single-dimensional array of Scalars (pointer to Scalar)
  ('shape', ctypes.POINTER(ctypes.c_int)),     # Pointer to array holding dimensions of the tensor
  ('strides', ctypes.POINTER(ctypes.c_int)),   # Pointer to array of strides
  ('ndim', ctypes.c_int),                      # Number of dimensions in the tensor
  ('size', ctypes.c_int),                      # Total number of elements in the tensor
  ('dtype', ctypes.c_int),                     # Data type of the tensor
  ('_prev', ctypes.POINTER(ctypes.POINTER(CTensor))),  # Track previous Tensors for autograd
  ('_prev_size', ctypes.c_int),                # Number of previous Tensors
]

lib.initialize_tensor.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.initialize_tensor.restype = ctypes.POINTER(CTensor)

lib.calculate_strides.argtypes = [ctypes.POINTER(CTensor)]
lib.calculate_strides.restype = None

lib.get_offset.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(CTensor)]
lib.get_offset.restype = ctypes.c_int

lib.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.delete_tensor.restype = None

lib.add_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.add_tensor.restype = ctypes.POINTER(CTensor)

lib.mul_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.mul_tensor.restype = ctypes.POINTER(CTensor)

lib.sub_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
lib.sub_tensor.restype = ctypes.POINTER(CTensor)

lib.neg_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.neg_tensor.restype = ctypes.POINTER(CTensor)

lib.pow_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
lib.pow_tensor.restype = ctypes.POINTER(CTensor)

lib.relu_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.relu_tensor.restype = ctypes.POINTER(CTensor)

lib.gelu_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.gelu_tensor.restype = ctypes.POINTER(CTensor)

lib.tanh_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.tanh_tensor.restype = ctypes.POINTER(CTensor)

lib.sigmoid_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.sigmoid_tensor.restype = ctypes.POINTER(CTensor)

lib.silu_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.silu_tensor.restype = ctypes.POINTER(CTensor)

lib.swiglu_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.swiglu_tensor.restype = ctypes.POINTER(CTensor)

lib.backward_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.backward_tensor.restype = None

lib.get_tensor_data.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
lib.get_tensor_data.restype = ctypes.c_double

lib.get_tensor_grad.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]
lib.get_tensor_grad.restype = ctypes.c_double

lib.set_tensor_data.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_double]
lib.set_tensor_data.restype = None

lib.set_tensor_grad.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_double]
lib.set_tensor_grad.restype = None

lib.print_tensor.argtypes = [ctypes.POINTER(CTensor)]
lib.print_tensor.restype = None