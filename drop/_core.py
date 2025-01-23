import ctypes, os
from typing import *
from ctypes import c_void_p, POINTER, c_int, c_float, CFUNCTYPE, c_double

DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_FLOAT32, DTYPE_FLOAT64 = 0, 1, 2, 3, 4, 5

libscalar_path = os.path.join(os.path.dirname(__file__), './build/libscalar.so')
libscalar = ctypes.CDLL(libscalar_path)
libtensor_path = os.path.join(os.path.dirname(__file__), './build/libtensor.so')
libtensor = ctypes.CDLL(libtensor_path)

class CScalar(ctypes.Structure):
  pass

CScalar._fields_ = [
  ('data', c_void_p),
  ('grad', c_void_p),
  ('dtype', c_int),
  ('_prev', POINTER(POINTER(CScalar))),
  ('_prev_size', c_int),
  ('_backward', CFUNCTYPE(None, POINTER(CScalar))),
  ('aux', c_float),
]

libscalar.initialize_scalars.argtypes = [c_float, c_int, POINTER(POINTER(CScalar)), c_int]
libscalar.initialize_scalars.restype = POINTER(CScalar)
libscalar.add_val.argtypes = [POINTER(CScalar), POINTER(CScalar)]
libscalar.add_val.restype = POINTER(CScalar)
libscalar.mul_val.argtypes = [POINTER(CScalar), POINTER(CScalar)]
libscalar.mul_val.restype = POINTER(CScalar)
libscalar.pow_val.argtypes = [POINTER(CScalar), c_float]
libscalar.pow_val.restype = POINTER(CScalar)
libscalar.negate.argtypes = [POINTER(CScalar)]
libscalar.negate.restype = POINTER(CScalar)
libscalar.sub_val.argtypes = [POINTER(CScalar), POINTER(CScalar)]
libscalar.sub_val.restype = POINTER(CScalar)
libscalar.div_val.argtypes = [POINTER(CScalar), POINTER(CScalar)]
libscalar.div_val.restype = POINTER(CScalar)
libscalar.log_val.argtypes = [POINTER(CScalar)]
libscalar.log_val.restype = POINTER(CScalar)
libscalar.relu.argtypes = [POINTER(CScalar)]
libscalar.relu.restype = POINTER(CScalar)
libscalar.sigmoid.argtypes = [POINTER(CScalar)]
libscalar.sigmoid.restype = POINTER(CScalar)
libscalar.tan_h.argtypes = [POINTER(CScalar)]
libscalar.tan_h.restype = POINTER(CScalar)
libscalar.gelu.argtypes = [POINTER(CScalar)]
libscalar.gelu.restype = POINTER(CScalar)
libscalar.silu.argtypes = [POINTER(CScalar)]
libscalar.silu.restype = POINTER(CScalar)
libscalar.swiglu.argtypes = [POINTER(CScalar)]
libscalar.swiglu.restype = POINTER(CScalar)
libscalar.backward.argtypes = [POINTER(CScalar)]
libscalar.backward.restype = None
libscalar.print.argtypes = [POINTER(CScalar)]
libscalar.print.restype = None
libscalar.cleanup.argtypes = [POINTER(CScalar)]
libscalar.cleanup.restype = None
libscalar.get_scalar_data.argtypes = [POINTER(CScalar)]
libscalar.get_scalar_data.restype = c_float
libscalar.get_scalar_grad.argtypes = [POINTER(CScalar)]
libscalar.get_scalar_grad.restype = c_float
libscalar.set_scalar_data.argtypes = [POINTER(CScalar), c_float]
libscalar.set_scalar_data.restype = None
libscalar.set_scalar_grad.argtypes = [POINTER(CScalar), c_float]
libscalar.set_scalar_grad.restype = None

class CTensor(ctypes.Structure):
  pass

CTensor._fields_ = [
  ('data', POINTER(c_void_p)),   # Single-dimensional array of Scalars (pointer to Scalar)
  ('shape', POINTER(c_int)),     # Pointer to array holding dimensions of the tensor
  ('strides', POINTER(c_int)),   # Pointer to array of strides
  ('ndim', c_int),                      # Number of dimensions in the tensor
  ('size', c_int),                      # Total number of elements in the tensor
  ('dtype', c_int),                     # Data type of the tensor
  ('_prev', POINTER(POINTER(CTensor))),  # Track previous Tensors for autograd
  ('_prev_size', c_int),                # Number of previous Tensors
]

libtensor.initialize_tensor.argtypes = [POINTER(c_double), c_int, POINTER(c_int), c_int]
libtensor.initialize_tensor.restype = POINTER(CTensor)
libtensor.calculate_strides.argtypes = [POINTER(CTensor)]
libtensor.calculate_strides.restype = None
libtensor.get_offset.argtypes = [POINTER(c_int), POINTER(CTensor)]
libtensor.get_offset.restype = c_int
libtensor.delete_tensor.argtypes = [POINTER(CTensor)]
libtensor.delete_tensor.restype = None
libtensor.add_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.add_tensor.restype = POINTER(CTensor)
libtensor.mul_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.mul_tensor.restype = POINTER(CTensor)
libtensor.sub_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.sub_tensor.restype = POINTER(CTensor)
libtensor.neg_tensor.argtypes = [POINTER(CTensor)]
libtensor.neg_tensor.restype = POINTER(CTensor)
libtensor.pow_tensor.argtypes = [POINTER(CTensor), c_float]
libtensor.pow_tensor.restype = POINTER(CTensor)
libtensor.relu_tensor.argtypes = [POINTER(CTensor)]
libtensor.relu_tensor.restype = POINTER(CTensor)
libtensor.gelu_tensor.argtypes = [POINTER(CTensor)]
libtensor.gelu_tensor.restype = POINTER(CTensor)
libtensor.tanh_tensor.argtypes = [POINTER(CTensor)]
libtensor.tanh_tensor.restype = POINTER(CTensor)
libtensor.sigmoid_tensor.argtypes = [POINTER(CTensor)]
libtensor.sigmoid_tensor.restype = POINTER(CTensor)
libtensor.silu_tensor.argtypes = [POINTER(CTensor)]
libtensor.silu_tensor.restype = POINTER(CTensor)
libtensor.swiglu_tensor.argtypes = [POINTER(CTensor)]
libtensor.swiglu_tensor.restype = POINTER(CTensor)
libtensor.backward_tensor.argtypes = [POINTER(CTensor)]
libtensor.backward_tensor.restype = None
# libtensor.get_tensor_data.argtypes = [POINTER(CTensor), c_int]
# libtensor.get_tensor_data.restype = c_double
# libtensor.get_tensor_grad.argtypes = [POINTER(CTensor), c_int]
# libtensor.get_tensor_grad.restype = c_double
# libtensor.set_tensor_data.argtypes = [POINTER(CTensor), c_int, c_double]
# libtensor.set_tensor_data.restype = None
# libtensor.set_tensor_grad.argtypes = [POINTER(CTensor), c_int, c_double]
# libtensor.set_tensor_grad.restype = None
# libtensor.print_tensor.argtypes = [POINTER(CTensor)]
# libtensor.print_tensor.restype = None