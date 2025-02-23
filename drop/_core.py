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