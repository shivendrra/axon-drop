#ifndef CPU_H
#define CPU_H

#include "dtype.h"
#include "tensor.h"
#include "scalar.h"

void add_tensor_cpu(Tensor* a, Tensor* b, float* out);
void sub_tensor_cpu(Tensor* a, Tensor* b, float* out);
void mul_tensor_cpu(Tensor* a, Tensor* b, float* out);
void div_tensor_cpu(Tensor* a, Tensor* b, float* out);

void scalar_mul_tensor_cpu(Tensor* a, Scalar* b, float* out);
void scalar_div_tensor_cpu(Tensor* a, Scalar* b, float* out);
void tensor_div_scalar_cpu(Tensor* a, Scalar* b, float* out);
void sigmoid_tensor_cpu(Tensor* a, float* out);
void gelu_tensor_cpu(Tensor* a, float* out);
void tanh_tensor_cpu(Tensor* a, float* out);

#endif