#ifndef __CPU__H__
#define __CPU__H__

#include "scalar.h"
#include "tensor.h"
#include "dtype.h"

extern "C" {
  void add_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void sub_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void mul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void div_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);
  void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);
  void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);
  void div_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);
  void scalar_mul_tensor_cpu(Tensor* a, Scalar* b, Tensor* out);
  void scalar_div_tensor_cpu(Tensor* a, Scalar* b, Tensor* out);
  void tensor_div_scalar_cpu(Tensor* a, Scalar* b, Tensor* out);
  void scalar_pow_tensor_cpu(Tensor* a, Scalar* b, Tensor* out);
  void tensor_pow_scalar_cpu(Tensor* a, Scalar* exp, Tensor* out);
  void matmul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void broadcasted_matmul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);
  void batched_matmul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void ones_like_tensor_cpu(int size, Tensor* out);
  void zeros_like_tensor_cpu(int size, Tensor* out);
  void transpose_1d_tensor_cpu(Tensor* a, Tensor* out);
  void transpose_2d_tensor_cpu(Tensor* a, Tensor* out);
  void transpose_3d_tensor_cpu(Tensor* a, Tensor* out);
  void reassign_tensor_cpu(Tensor* a, Tensor* out);
  void make_contiguous_tensor_cpu(Tensor* a, Tensor* out);
  void equal_tensor_cpu(Tensor* a, Tensor* b, Tensor* out);
  void equal_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size);

  void log_tensor_cpu(Tensor* a, Tensor* out);
  void sum_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis);
  void max_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis);
  void min_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis);

  void sin_tensor_cpu(Tensor* a, Tensor* out);
  void cos_tensor_cpu(Tensor* a, Tensor* out);
  void sigmoid_tensor_cpu(Tensor* a, Tensor* out);
  void tanh_tensor_cpu(Tensor* a, Tensor* out);
  void relu_tensor_cpu(Tensor* a, Tensor* out);
  void gelu_tensor_cpu(Tensor* a, Tensor* out);
  void swiglu_tensor_cpu(Tensor* a, Tensor* out);
  void silu_tensor_cpu(Tensor* a, Tensor* out);
}

#endif