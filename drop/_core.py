import ctypes, os
from typing import *
from ctypes import c_void_p, POINTER, c_int, c_float, CFUNCTYPE, c_bool, c_char_p

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
  ('data', POINTER(CScalar)),
  ('dtype', c_int),
  ('strides', POINTER(c_int)),
  ('backstrides', POINTER(c_int)),
  ('shape', POINTER(c_int)),
  ('size', c_int),
  ('ndim', c_int),
]

libtensor.create_tensor.argtypes = [POINTER(c_float), POINTER(c_int), c_int, c_int]
libtensor.create_tensor.restype = POINTER(CTensor)
libtensor.delete_tensor.argtypes = [POINTER(CTensor)]
libtensor.delete_tensor.restype = None
libtensor.delete_strides.argtypes = [POINTER(CTensor)]
libtensor.delete_strides.restype = None
libtensor.delete_shape.argtypes = [POINTER(CTensor)]
libtensor.delete_shape.restype = None
libtensor.delete_data.argtypes = [POINTER(CTensor)]
libtensor.delete_data.restype = None
libtensor.delete_backstrides.argtypes = [POINTER(CTensor)]
libtensor.delete_backstrides.restype = None
libtensor.add_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.add_tensor.restype = POINTER(CTensor)
libtensor.sub_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.sub_tensor.restype = POINTER(CTensor)
libtensor.elemwise_mul_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.elemwise_mul_tensor.restype = POINTER(CTensor)
libtensor.tensor_div_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.tensor_div_tensor.restype = POINTER(CTensor)
libtensor.add_broadcasted_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.add_broadcasted_tensor.restype = POINTER(CTensor)
libtensor.sub_broadcasted_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.sub_broadcasted_tensor.restype = POINTER(CTensor)
libtensor.elemwise_mul_broadcasted_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.elemwise_mul_broadcasted_tensor.restype = POINTER(CTensor)
libtensor.matmul_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.matmul_tensor.restype = POINTER(CTensor)
libtensor.batched_matmul_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.batched_matmul_tensor.restype = POINTER(CTensor)
libtensor.broadcasted_batched_matmul_tensor_cpu.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.broadcasted_batched_matmul_tensor_cpu.restype = POINTER(CTensor)
libtensor.scalar_mul_tensor.argtypes = [POINTER(CTensor), POINTER(CScalar)]
libtensor.scalar_mul_tensor.restype = POINTER(CTensor)
libtensor.scalar_div_tensor.argtypes = [POINTER(CScalar), POINTER(CTensor)]
libtensor.scalar_div_tensor.restype = POINTER(CTensor)
libtensor.tensor_div_scalar.argtypes = [POINTER(CTensor), POINTER(CScalar)]
libtensor.tensor_div_scalar.restype = POINTER(CTensor)
libtensor.tensor_pow_scalar.argtypes = [POINTER(CTensor), POINTER(CScalar)]
libtensor.tensor_pow_scalar.restype = POINTER(CTensor)
libtensor.scalar_pow_tensor.argtypes = [POINTER(CScalar), POINTER(CTensor)]
libtensor.scalar_pow_tensor.restype = POINTER(CTensor)
libtensor.log_tensor.argtypes = [POINTER(CTensor)]
libtensor.log_tensor.restype = POINTER(CTensor)
libtensor.sum_tensor.argtypes = [POINTER(CTensor), c_int, c_bool]
libtensor.sum_tensor.restype = POINTER(CTensor)
libtensor.max_tensor.argtypes = [POINTER(CTensor), c_int, c_bool]
libtensor.max_tensor.restype = POINTER(CTensor)
libtensor.min_tensor.argtypes = [POINTER(CTensor), c_int, c_bool]
libtensor.min_tensor.restype = POINTER(CTensor)
libtensor.sigmoid_tensor.argtypes = [POINTER(CTensor)]
libtensor.sigmoid_tensor.restype = POINTER(CTensor)
libtensor.sin_tensor.argtypes = [POINTER(CTensor)]
libtensor.sin_tensor.restype = POINTER(CTensor)
libtensor.cos_tensor.argtypes = [POINTER(CTensor)]
libtensor.cos_tensor.restype = POINTER(CTensor)
libtensor.tanh_tensor.argtypes = [POINTER(CTensor)]
libtensor.tanh_tensor.restype = POINTER(CTensor)
libtensor.relu_tensor.argtypes = [POINTER(CTensor)]
libtensor.relu_tensor.restype = POINTER(CTensor)
libtensor.gelu_tensor.argtypes = [POINTER(CTensor)]
libtensor.gelu_tensor.restype = POINTER(CTensor)
libtensor.swiglu_tensor.argtypes = [POINTER(CTensor)]
libtensor.swiglu_tensor.restype = POINTER(CTensor)
libtensor.silu_tensor.argtypes = [POINTER(CTensor)]
libtensor.silu_tensor.restype = POINTER(CTensor)
libtensor.reshape_tensor.argtypes = [POINTER(CTensor), POINTER(c_int), c_int]
libtensor.reshape_tensor.restype = POINTER(CTensor)
libtensor.transpose_tensor.argtypes = [POINTER(CTensor)]
libtensor.transpose_tensor.restype = POINTER(CTensor)
libtensor.make_contiguous.argtypes = [POINTER(CTensor)]
libtensor.make_contiguous.restype = None
libtensor.equal_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.equal_tensor.restype = POINTER(CTensor)
libtensor.equal_broadcasted_tensor.argtypes = [POINTER(CTensor), POINTER(CTensor)]
libtensor.equal_broadcasted_tensor.restype = POINTER(CTensor)
libtensor.zeros_like_tensor.argtypes = [POINTER(CTensor)]
libtensor.zeros_like_tensor.restype = POINTER(CTensor)
libtensor.ones_like_tensor.argtypes = [POINTER(CTensor)]
libtensor.ones_like_tensor.restype = POINTER(CTensor)
libtensor.print_tensor.argtypes = [POINTER(CTensor)]
libtensor.print_tensor.restype = None
libtensor.tensor_backward.argtypes = [POINTER(CTensor)]
libtensor.tensor_backward.restype = None
libtensor.print_grads.argtypes = [POINTER(CTensor)]
libtensor.print_grads.restype = None