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
    Scalar* s_out = *add_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void sub_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size, i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = *sub_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void mul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size, i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = *mul_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void div_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for int(i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data[i];
    Scalar* s_out = *div_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    Scalar* s_a = &a->data[index1];
    Scalar* s_b = &b->data[index2];
    Scalar* s_out = *add_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    Scalar* s_a = &a->data[index1];
    Scalar* s_b = &b->data[index2];
    Scalar* s_out = *sub_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    Scalar* s_a = &a->data[index1];
    Scalar* s_b = &b->data[index2];
    Scalar* s_out = *mul_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void div_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    Scalar* s_a = &a->data[index1];
    Scalar* s_b = &b->data[index2];
    Scalar* s_out = *div_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void scalar_mul_tensor_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data;
    Scalar* s_out = *mul_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void scalar_div_tensor_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data;
    Scalar* s_out = *div_val(s_b, s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tensor_div_scalar_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data;
    Scalar* s_out = *div_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void scalar_pow_tensor_cpu(Tensor* base, Scalar a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data;
    Scalar* s_out = *pow_val(s_b, s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tensor_pow_scalar_cpu(Tensor* a, Scalar exp, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &exp->data;
    Scalar* s_out = *pow_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void log_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *log_val(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tanh_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *tan_h(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void sigmoid_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *sigmoid(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void relu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *relu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void gelu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *gelu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void swiglu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *swiglu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void silu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *silu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void equal_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for int(i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data[i];
    Scalar* s_out = *equal_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void equal_broadcasted_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    Scalar* s_a = &a->data[index1];
    Scalar* s_b = &b->data[index2];
    Scalar* s_out = *equal_val(s_a, s_b);
    out[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void reassign_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = *initialize_scalars(get_data_as_float(s_a, s_a->dtype), a->dtype, a->_prev, a->_prev_size);
    out[i] = *s_a;
    free(s_a);
  }
}

void zeros_like_tensor_cpu(int a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_out = *initialize_scalars(0.0f, a->dtype, NULL, 0);
    out[i] = *s_a;
    free(s_a);
  }
}

void ones_like_tensor_cpu(int a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_out = *initialize_scalars(1.0f, a->dtype, NULL, 0);
    out[i] = *s_a;
    free(s_a);
  }
}