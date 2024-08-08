import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'libscalar.so')
lib = ctypes.CDLL(lib_path)

class CScalar(ctypes.Structure):
  pass

CScalar._fields_ = [
  ('data', ctypes.c_double),
  ('grad', ctypes.c_double),
  ('_prev', ctypes.POINTER(ctypes.POINTER(CScalar))),
  ('_prev_size', ctypes.c_int),
  ('_backward', ctypes.CFUNCTYPE(None, ctypes.POINTER(CScalar))),
  ('aux', ctypes.c_double),
]

lib.initialize_scalars.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.POINTER(CScalar)), ctypes.c_int]
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

lib.backward.argtypes = [ctypes.POINTER(CScalar)]
lib.backward.restype = None

lib.print.argtypes = [ctypes.POINTER(CScalar)]
lib.print.restype = None

lib.get_scalar_data.argtypes = [ctypes.POINTER(CScalar)]
lib.get_scalar_data.restype = ctypes.c_double

lib.get_scalar_grad.argtypes = [ctypes.POINTER(CScalar)]
lib.get_scalar_grad.restype = ctypes.c_double