from setuptools import setup, find_packages
import os
import codecs

current_dir = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(current_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

VERSION = '0.0.9'
DESCRIPTION = ('Tensor manipulation library wrapped over a scalar-level autograd '
               'to compute backpropagation like PyTorch, but more like microgad')

# Paths for the shared libraries (inside drop/build)
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

setup(
  name="axon-drop",
  version=VERSION,
  author="shivendra",
  author_email="shivharsh44@gmail.com",
  description=DESCRIPTION,
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="MIT",
  packages=find_packages(include=['drop', 'drop.*']),
  classifiers=[
      "Development Status :: 2 - Pre-Alpha",
      "Intended Audience :: Developers",
      "Intended Audience :: Education",
      "Programming Language :: C",
      "Programming Language :: C++",
      "Programming Language :: Python :: 3.11",
      "Operating System :: OS Independent",
      "License :: OSI Approved :: MIT License",
  ],
  # Include the shared libraries (.so files) from drop/build and C++ sources from csrc/
  package_data={
      'drop': [
          'build/libscalar.so',
          'build/libtensor.so',
      ],
      'csrc': [
          '*.h', 
          '*.cpp'
      ]
  },
  include_package_data=True,
  entry_points={
      'console_scripts': [
          'drop=drop.__main__:main',
      ],
  }
)