#include "scalar.h"
#include "dtype.h"
#include "tensor.h"
#include "cpu.h"
#include <stdio.h>

void add_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size, i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    
    // perform Scalar ops while preserving prev Scalars for autograd
    // & free the data after replacing the new output with original sample scalar
    // values set in dummy ``out`` tensor
    Scalar* s_out = add_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void sub_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size, i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = sub_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void mul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size, i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = mul_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}