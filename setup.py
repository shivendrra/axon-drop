import os
import setuptools

# Determine the project root.
current_dir = os.path.abspath(os.path.dirname(__file__))

# Check for required shared libraries in drop/build
libscalar_path = os.path.join(current_dir, 'drop', 'build', 'libscalar.so')
libtensor_path = os.path.join(current_dir, 'drop', 'build', 'libtensor.so')

if not os.path.exists(libscalar_path):
  raise FileNotFoundError(
    f"Shared library {libscalar_path} not found. Please compile the C++ code to generate libscalar.so"
  )
if not os.path.exists(libtensor_path):
  raise FileNotFoundError(
    f"Shared library {libtensor_path} not found. Please compile the C++ code to generate libtensor.so"
  )

setuptools.setup()