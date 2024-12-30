/* 
  - tensor.h header file for tensor.cpp & Tensor
  - contains the wrapper over Scalar values & computes autograd for each value
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libtensor.so tensor.cpp cpu.cpp dtype.cpp scalar.cpp
    -- '.dll': g++ -shared -o libtensor.dll tensor.cpp cpu.cpp dtype.cpp scalar.cpp
*/

#ifndef __TENSOR__H__
#define __TENSOR__H__

#include "scalar.h"
#include "dtype.h"
#include <stdio.h>

typedef struct {
  Scalar* data;                 // 1-d array containing bunch of Scalar values
  DType dtype;                  // target dtype for the Tensor
  int* strides;                 // strides for reshaping & other ops
  int* backstrides;             // reversed strides for reshaping, transposing
  int* shape;                   // array for defining shape of tensor
  int size;                     // total no of elements
  int ndim;                     // no of dims
  char* device;                 // device for ops
} Tensor;

extern "C" {
  Tensor* create_tensor(float* data, int* shape, int ndim, char* device, DType dtype);
  void to_device(Tensor* a, char* device);
  void delete_tensor(Tensor* tensor);
  void delete_strides(Tensor* tensor);
  void delete_shape(Tensor* tensor);
  void delete_data(Tensor* tensor);
  void delete_device(Tensor* tensor);
  void delete_backstrides(Tensor* tensor);

  Tensor* add_tensor(Tensor* a, Tensor* b);
  Tensor* sub_tensor(Tensor* a, Tensor* b);
  Tensor* elemwise_mul_tensor(Tensor* a, Tensor* b);
  Tensor* tensor_div_tensor(Tensor* a, Tensor* b);
  Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* elemwise_mul_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* matmul_tensor(Tensor* a, Tensor* b);
  Tensor* batched_matmul_tensor(Tensor* a, Tensor* b);
  Tensor* broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b);
  Tensor* scalar_mul_tensor(Tensor* a, float b);
  Tensor* scalar_div_tensor(float a, Tensor* b);
  Tensor* tensor_div_scalar(Tensor* a, float b);
  Tensor* tensor_pow_scalar(Tensor* a, float exponent);
  Tensor* scalar_pow_tensor(float base, Tensor* a);
  Tensor* log_tensor(Tensor* a);
  Tensor* sum_tensor(Tensor* a, int axis, bool keepdim);
  Tensor* max_tensor(Tensor* a, int axis, bool keepdim);
  Tensor* min_tensor(Tensor* a, int axis, bool keepdim);
  Tensor* sigmoid_tensor(Tensor* a);
  Tensor* tanh_tensor(Tensor* a);
  Tensor* relu_tensor(Tensor* a);

  Tensor* reshape_tensor(Tensor* a, int* new_shape, int new_ndim);
  Tensor* transpose_tensor(Tensor* a);
  void make_contiguous(Tensor* a);
  Tensor* equal_tensor(Tensor* a, Tensor* b);
  Tensor* equal_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* zeros_like_tensor(Tensor* a);
  Tensor* ones_like_tensor(Tensor* a);
}

#endif