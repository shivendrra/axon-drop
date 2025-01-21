#include "scalar.h"
#include "dtype.h"
#include "tensor.h"
#include "cpu.h"
#include <stdio.h>
#include <cstring>

void add_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size; i++) {
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
  for (int i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = sub_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void mul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];  // Scalar value from a
    Scalar* s_b = &b->data[i];  // Scalar value from b
    Scalar* s_out = mul_val(s_a, s_b);
    out->data[i] = *s_out;  // assign the value to output Tensor
    free(s_out);
  }
}

void div_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data[i];
    Scalar* s_out = div_val(s_a, s_b);
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
    Scalar* s_out = add_val(s_a, s_b);
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
    Scalar* s_out = sub_val(s_a, s_b);
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
    Scalar* s_out = mul_val(s_a, s_b);
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
    Scalar* s_out = div_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void scalar_mul_tensor_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b;
    Scalar* s_out = mul_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void scalar_div_tensor_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b;
    Scalar* s_out = div_val(s_b, s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tensor_div_scalar_cpu(Tensor* a, Scalar b, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b;
    Scalar* s_out = div_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void scalar_pow_tensor_cpu(Tensor* base, Scalar* a, Tensor* out) {
  for (int i = 0; i < base->size; i++) {
    Scalar* s_a = &base->data[i];
    float s_b = get_data_as_float(a->data, a->dtype);
    Scalar* s_out = pow_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tensor_pow_scalar_cpu(Tensor* a, Scalar* exp, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    float s_b = get_data_as_float(exp->data, exp->dtype);
    Scalar* s_out = pow_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void reassign_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    out->data[i] = a->data[i];
  }
}

void make_contiguous_tensor_cpu(Tensor* a, Tensor* out) {
  if (!a || !out) {
    fprintf(stderr, "Null Tensor provided for make_contiguous_tensor_cpu.\n");
    return;
  }
  out->shape = (int*)malloc(a->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Failed to allocate memory for output tensor shape.\n");
    exit(1);
  }
  memcpy(out->shape, a->shape, a->ndim * sizeof(int));
  out->ndim = a->ndim;
  out->size = a->size;

  // recalculate strides and backstrides for the contiguous layout
  out->strides = (int*)malloc(a->ndim * sizeof(int));
  if (!out->strides) {
    fprintf(stderr, "Failed to allocate memory for output tensor strides.\n");
    exit(1);
  }
  out->backstrides = (int*)malloc(a->ndim * sizeof(int));
  if (!out->backstrides) {
    fprintf(stderr, "Failed to allocate memory for output tensor backstrides.\n");
    exit(1);
  }

  int stride = 1;
  for (int i = a->ndim - 1; i >= 0; i--) {
    out->strides[i] = stride;
    out->backstrides[i] = (a->shape[i] - 1) * stride;
    stride *= a->shape[i];
  }

  // allocating memory for the contiguous data
  out->data = (Scalar*)malloc(a->size * sizeof(Scalar));
  if (!out->data) {
    fprintf(stderr, "Failed to allocate memory for output tensor data.\n");
    exit(1);
  }

  out->dtype = a->dtype;
  out->device = (char*)malloc(strlen(a->device) + 1);
  if (!out->device) {
    fprintf(stderr, "Failed to allocate memory for output tensor device.\n");
    exit(1);
  }
  strcpy(out->device, a->device);

  // initializing the output tensor's data with Scalars compatible with autograd
  for (int i = 0; i < a->size; i++) {
    // calculating source index in the original tensor
    int source_index = 0;
    int linear_index = i;
    for (int j = a->ndim - 1; j >= 0; j--) {
      int pos = linear_index % a->shape[j];
      linear_index /= a->shape[j];
      source_index += pos * a->strides[j];
    }

    // copy & link Scalars for autograd compatibility
    Scalar* parent_scalar = &a->data[source_index]; // Create a Scalar* for the parent
    Scalar* temp_scalar = initialize_scalars(get_scalar_data(parent_scalar), a->dtype, &parent_scalar, 1);
    out->data[i] = *temp_scalar;
    cleanup(temp_scalar);
  }
}

void log_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = log_val(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void tanh_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = tan_h(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void sigmoid_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = sigmoid(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void relu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = relu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void gelu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = gelu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void swiglu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = swiglu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void silu_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = silu(s_a);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void equal_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i <= a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_b = &b->data[i];
    Scalar* s_out = equal_val(s_a, s_b);
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
    Scalar* s_out = equal_val(s_a, s_b);
    out->data[i] = *s_out;
    free(s_out);
  }
  free(strides1);
  free(strides2);
}

void reassign_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    Scalar* s_a = &a->data[i];
    Scalar* s_out = initialize_scalars(get_data_as_float(s_a, s_a->dtype), a->dtype, s_a->_prev, s_a->_prev_size);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void zeros_like_tensor_cpu(int size, Tensor* out) {
  for (int i = 0; i < size; i++) {
    Scalar* s_out = initialize_scalars(0.0f, out->dtype, NULL, 0);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void ones_like_tensor_cpu(int size, Tensor* out) {
  for (int i = 0; i < size; i++) {
    Scalar* s_out = initialize_scalars(1.0f, out->dtype, NULL, 0);
    out->data[i] = *s_out;
    free(s_out);
  }
}

void mamtul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int i = 0; i < a->shape[0]; i++) {
    for (int j = 0; j < b->shape[1]; j++) {
      Scalar* sum = initialize_scalars(0.0f, a->dtype, NULL, 0);
      for (int k = 0; k < a->shape[1]; k++) {
        Scalar* s_a = &a->data[i * a->shape[1] + k];
        Scalar* s_b = &b->data[k * b->shape[1] + j];
        Scalar* product = mul_val(s_a, s_b);
        Scalar* updated_sum = add_val(sum, product);
        cleanup(sum);
        cleanup(product);
        sum = updated_sum;
      }
      out->data[i * b->shape[1] + j] = *sum;
      cleanup(sum);
    }
  }
}

void broadcasted_matmul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out, int* broadcasted_shape, int broadcasted_size) {
  int broadcast_dims = broadcasted_size / (a->shape[0] * b->shape[1]);
  for (int batch = 0; batch < broadcast_dims; batch++) {
    Tensor a_sub;
    a_sub.ndim = a->ndim, a_sub.data = &a->data[batch * a->size], a_sub.shape = a->shape, a_sub.dtype = a->dtype;
    Tensor b_sub;
    b_sub.ndim = b->ndim, b_sub.data = &b->data[batch * b->size], b_sub.shape = b->shape, b_sub.dtype = b->dtype;
    Tensor out_sub;
    out_sub.ndim = out->ndim, out_sub.data = &out->data[batch * a->shape[0] * b->shape[1]], out_sub.shape = out->shape, out_sub.dtype = out->dtype;

    matmul_tensor_cpu(&a_sub, &b_sub, &out_sub);
  }
}

void batched_matmul_tensor_cpu(Tensor* a, Tensor* b, Tensor* out) {
  for (int batch = 0; batch < a->shape[0]; batch++) {
    Tensor a_matrix;
    a_matrix.data = &a->data[batch * a->shape[1] * a->shape[2]], a_matrix.shape = &a->shape[1], a_matrix.ndim = 2, a_matrix.dtype = a->dtype;
    Tensor b_matrix;
    b_matrix.data = &b->data[batch * b->shape[1] * b->shape[2]], b_matrix.shape = &b->shape[1], b_matrix.ndim = 2, b_matrix.dtype = b->dtype;
    Tensor out_matrix;
    out_matrix.data = &out->data[batch * a->shape[1] * b->shape[2]], out_matrix.shape = &out->shape[1], out_matrix.ndim = 2, out_matrix.dtype = out->dtype;

    matmul_tensor_cpu(&a_matrix, &b_matrix, &out_matrix);
  }
}

void transpose_1d_tensor_cpu(Tensor* a, Tensor* out) {
  for (int i = 0; i < a->size; i++) {
    out->data[i] = a->data[i];  // simply copying scalars
  }
  out->shape[0] = a->shape[0];
  out->ndim = 1;
  out->dtype = a->dtype;
  out->device = strdup(a->device);
}

void transpose_2d_tensor_cpu(Tensor* a, Tensor* out) {
  int rows = a->shape[0], cols = a->shape[1];
  out->shape[0] = cols;
  out->shape[1] = rows;
  out->ndim = 2;
  out->dtype = a->dtype;
  out->device = strdup(a->device);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int in_index = i * cols + j;
      int out_index = j * rows + i;
      out->data[out_index] = a->data[in_index];  // copy Scalars
    }
  }
}

void transpose_3d_tensor_cpu(Tensor* a, Tensor* out, int axis1, int axis2) {
  // compute the new shape by swapping the specified axes
  int new_shape[3] = {a->shape[0], a->shape[1], a->shape[2]};
  new_shape[axis1] = a->shape[axis2];
  new_shape[axis2] = a->shape[axis1];

  // updated output tensor properties
  memcpy(out->shape, new_shape, 3 * sizeof(int));
  out->ndim = 3;
  out->dtype = a->dtype;
  out->device = strdup(a->device);

  // iterating through the 3D tensor & transpose specified axes
  for (int i = 0; i < new_shape[0]; i++) {
    for (int j = 0; j < new_shape[1]; j++) {
      for (int k = 0; k < new_shape[2]; k++) {
        // computing the input indices based on transpose mapping
        int indices[3] = {i, j, k};
        int temp = indices[axis1];
        indices[axis1] = indices[axis2];
        indices[axis2] = temp;

        int in_index = indices[0] * (a->shape[1] * a->shape[2]) +
                       indices[1] * a->shape[2] + indices[2];
        int out_index = i * (new_shape[1] * new_shape[2]) +
                        j * new_shape[2] + k;

        out->data[out_index] = a->data[in_index];  // copy Scalars
      }
    }
  }
}

void max_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    // compute max over all elements without picking out each data element
    Scalar* max_scalar = initialize_scalars(get_scalar_data(&a->data[0]), a->dtype, NULL, 0);
    for (int i = 1; i < a->size; i++) {
      if (get_scalar_data(&a->data[i]) > get_scalar_data(max_scalar)) {
        cleanup(max_scalar);  // free the old max_scalar
        max_scalar = initialize_scalars(get_scalar_data(&a->data[i]), a->dtype, NULL, 0);
      }
    }
    out->data[0] = *max_scalar;  // assign the final max Scalar
  } else {
    // compute max along a specific axis
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "Invalid axis specified for max operation.\n");
      return;
    }

    int axis_stride = a->strides[axis];
    for (int i = 0; i < size; i++) {
      Scalar* max_scalar = initialize_scalars(get_scalar_data(&a->data[0]), a->dtype, NULL, 0);
      for (int j = 1; j < a->shape[axis]; j++) {
        int offset = j * axis_stride;
        int index = 0;
        int remainder = i;

        for (int k = a->ndim - 1; k >= 0; k--) {
          if (k != axis) {
            index += (remainder % res_shape[k]) * a->strides[k];
            remainder /= res_shape[k];
          }
        }

        if (get_scalar_data(&a->data[index + offset]) > get_scalar_data(max_scalar)) {
          cleanup(max_scalar);  // free the old max_scalar
          max_scalar = initialize_scalars(get_scalar_data(&a->data[index + offset]), a->dtype, NULL, 0);
        }
      }
      out->data[i] = *max_scalar;  // assign the final max Scalar
    }
  }
}

void min_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    // compute min over all elements same as max_tensor_cpu
    Scalar* min_scalar = initialize_scalars(get_scalar_data(&a->data[0]), a->dtype, NULL, 0);
    for (int i = 1; i < a->size; i++) {
      if (get_scalar_data(&a->data[i]) < get_scalar_data(min_scalar)) {
        cleanup(min_scalar);  // free the old min_scalar
        min_scalar = initialize_scalars(get_scalar_data(&a->data[i]), a->dtype, NULL, 0);
      }
    }
    out->data[0] = *min_scalar;  // assign the final min Scalar
  } else {
    // compute min along a specific axis
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "Invalid axis specified for min operation.\n");
      exit(EXIT_FAILURE);
    }

    int axis_stride = a->strides[axis];
    for (int i = 0; i < size; i++) {
      Scalar* min_scalar = initialize_scalars(get_scalar_data(&a->data[0]), a->dtype, NULL, 0);
      for (int j = 1; j < a->shape[axis]; j++) {
        int offset = j * axis_stride;
        int index = 0;
        int remainder = i;

        for (int k = a->ndim - 1; k >= 0; k--) {
          if (k != axis) {
            index += (remainder % res_shape[k]) * a->strides[k];
            remainder /= res_shape[k];
          }
        }

        if (get_scalar_data(&a->data[index + offset]) < get_scalar_data(min_scalar)) {
          cleanup(min_scalar);  // free the old min_scalar
          min_scalar = initialize_scalars(get_scalar_data(&a->data[index + offset]), a->dtype, NULL, 0);
        }
      }
      out->data[i] = *min_scalar;  // assign the final min Scalar
    }
  }
}

void sum_tensor_cpu(Tensor* a, Tensor* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    // compute sum over all elements by initiallizing buffer Zero-Scalar
    Scalar* sum = initialize_scalars(0.0f, a->dtype, NULL, 0);
    for (int i = 0; i < a->size; i++) {
      Scalar* new_sum = add_val(sum, &a->data[i]);
      cleanup(sum);  // free the previous sum to avoid memory leaks
      sum = new_sum;
    }
    out->data[0] = *sum;  // assign the final sum Scalar
  } else {
    // compute sum along a specific axis
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "Invalid axis specified for sum operation.\n");
      exit(EXIT_FAILURE);
    }

    int axis_stride = a->strides[axis];
    for (int i = 0; i < size; i++) {
      Scalar* sum = initialize_scalars(0.0f, a->dtype, NULL, 0);
      for (int j = 0; j < a->shape[axis]; j++) {
        int offset = j * axis_stride;
        int index = 0;
        int remainder = i;

        for (int k = a->ndim - 1; k >= 0; k--) {
          if (k != axis) {
            index += (remainder % res_shape[k]) * a->strides[k];
            remainder /= res_shape[k];
          }
        }

        Scalar* new_sum = add_val(sum, &a->data[index + offset]);
        cleanup(sum);  // free the previous sum to avoid memory leaks
        sum = new_sum;
      }
      out->data[i] = *sum;  // assign the final sum Scalar
    }
  }
}