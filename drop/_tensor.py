from .cbase import CScalar, lib
from .cbase import DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64, DTYPE_INT8
import ctypes
from typing import *

int8 = DTYPE_INT8
int16 = DTYPE_INT16
int32 = DTYPE_INT32
int64 = DTYPE_INT64
float32 = DTYPE_FLOAT32
float64 = DTYPE_FLOAT64

def initialize_tensor(data, dtype):
  pass