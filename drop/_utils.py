""" here lies the functions to generate sample tensors """

from .helpers.utils import _zeros, _ones, _randint, _randn, _arange, _zeros_like, _ones_like
from typing import *

def zeros(shape:tuple) -> list:
  return _zeros(shape)

def ones(shape:tuple) -> list:
  return _ones(shape)

def randint(low:int, high:int, size:int=None) -> list:
  return _randint(low, high, size)

def arange(start:int=0, end:int=10, step:int=1) -> list:
  return _arange(start, end, step)

def randn(domain:tuple=(1,-1), shape:tuple=None) -> list:
  return _randn(domain, shape)

def zeros_like(arr:list) -> list:
  return _zeros_like(arr)

def ones_like(arr:list) -> list:
  return _ones_like(arr)