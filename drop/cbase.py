import ctypes
import os

DTYPE_INT8 = 0
DTYPE_INT16 = 1
DTYPE_INT32 = 2
DTYPE_INT64 = 3
DTYPE_FLOAT32 = 4
DTYPE_FLOAT64 = 5

lib_path = os.path.join(os.path.dirname(__file__), './build/libscalar.so')
lib = ctypes.CDLL(lib_path)

class CScalar(ctypes.Structure):
  pass

CScalar._fields_ = [
  ('data', ctypes.c_void_p),
  ('grad', ctypes.c_void_p),
  ('dtype', ctypes.c_int),
  ('_prev', ctypes.POINTER(ctypes.POINTER(CScalar))),
  ('_prev_size', ctypes.c_int),
  ('_backward', ctypes.CFUNCTYPE(None, ctypes.POINTER(CScalar))),
  ('aux', ctypes.c_double),
]

lib.initialize_scalars.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(CScalar)), ctypes.c_int]
lib.initialize_scalars.restype = ctypes.POINTER(CScalar)

lib.add_val.argtypes = [ctypes.POINTER(CScalar), ctypes.POINTER(CScalar)]
lib.add_val.restype = ctypes.POINTER(CScalar)

lib.mul_val.argtypes = [ctypes.POINTER(CScalar), ctypes.POINTER(CScalar)]
lib.mul_val.restype = ctypes.POINTER(CScalar)

lib.pow_val.argtypes = [ctypes.POINTER(CScalar), ctypes.c_float]
lib.pow_val.restype = ctypes.POINTER(CScalar)

lib.negate.argtypes = [ctypes.POINTER(CScalar)]
lib.negate.restype = ctypes.POINTER(CScalar)

lib.sub_val.argtypes = [ctypes.POINTER(CScalar), ctypes.POINTER(CScalar)]
lib.sub_val.restype = ctypes.POINTER(CScalar)

lib.div_val.argtypes = [ctypes.POINTER(CScalar), ctypes.POINTER(CScalar)]
lib.div_val.restype = ctypes.POINTER(CScalar)

lib.relu.argtypes = [ctypes.POINTER(CScalar)]
lib.relu.restype = ctypes.POINTER(CScalar)

lib.sigmoid.argtypes = [ctypes.POINTER(CScalar)]
lib.sigmoid.restype = ctypes.POINTER(CScalar)

lib.tan_h.argtypes = [ctypes.POINTER(CScalar)]
lib.tan_h.restype = ctypes.POINTER(CScalar)

lib.gelu.argtypes = [ctypes.POINTER(CScalar)]
lib.gelu.restype = ctypes.POINTER(CScalar)

lib.silu.argtypes = [ctypes.POINTER(CScalar)]
lib.silu.restype = ctypes.POINTER(CScalar)

lib.swiglu.argtypes = [ctypes.POINTER(CScalar)]
lib.swiglu.restype = ctypes.POINTER(CScalar)

lib.backward.argtypes = [ctypes.POINTER(CScalar)]
lib.backward.restype = None

lib.print.argtypes = [ctypes.POINTER(CScalar)]
lib.print.restype = None

lib.cleanup.argtypes = [ctypes.POINTER(CScalar)]
lib.cleanup.restype = None

lib.get_scalar_data.argtypes = [ctypes.POINTER(CScalar)]
lib.get_scalar_data.restype = ctypes.c_double

lib.get_scalar_grad.argtypes = [ctypes.POINTER(CScalar)]
lib.get_scalar_grad.restype = ctypes.c_double

lib.set_scalar_data.argtypes = [ctypes.POINTER(CScalar), ctypes.c_double]
lib.set_scalar_data.restype = None

lib.set_scalar_grad.argtypes = [ctypes.POINTER(CScalar), ctypes.c_double]
lib.set_scalar_grad.restype = None

lib.add_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.add_backward.restype = None
lib.mul_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.mul_backward.restype = None
lib.pow_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.pow_backward.restype = None
lib.relu_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.relu_backward.restype = None
lib.tanh_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.tanh_backward.restype = None
lib.sigmoid_backward.argtypes = [ctypes.POINTER(CScalar)]
lib.sigmoid_backward.restype = None