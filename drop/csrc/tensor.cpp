#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "tensor.h"
#include "scalar.h"
#include "cpu.h"
#include "dtype.h"

Tensor* create_tensor(float* data, int* shape, int ndim, DType dtype) {
  Tensor* self = (Tensor*)malloc(sizeof(Tensor));
  if (!self) {
    fprintf(stderr, "Memory allocation failed for Tensor\n");
    exit(1);
  }
  self->shape = (int*)malloc(ndim * sizeof(int));
  if (!self->shape) {
    fprintf(stderr, "Memory allocation failed for Tensor shape\n");
    free(self);
    exit(1);
  }
  memcpy(self->shape, shape, ndim * sizeof(int));
  self->ndim = ndim;

  self->size = 1;
  for (int i = 0; i < ndim; i++) {
    self->size *= shape[i];
  }

  self->strides = (int*)malloc(ndim * sizeof(int));
  if (!self->strides) {
    fprintf(stderr, "Memory allocation failed for Tensor strides\n");
    free(self->shape);
    free(self);
    exit(1);
  }
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= shape[i];
  }
  self->backstrides = (int*)malloc(ndim * sizeof(int));
  if (!self->backstrides) {
    fprintf(stderr, "Memory allocation failed for Tensor backstrides\n");
    free(self->shape);
    free(self);
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    self->backstrides[i] = (shape[i] - 1) * self->strides[i];
  }
  self->aux = (float*)malloc(self->size * sizeof(float));
  if (!self->aux) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
  for (int i = 0; i < self->size; i++) {
    self->aux[i] = data[i];
  }
  // allocation memory for data (array of Scalars)
  self->data = (Scalar*)malloc(self->size * sizeof(Scalar));
  if (!self->data) {
    fprintf(stderr, "Memory allocation failed for Tensor data\n");
    free(self->strides);
    free(self->shape);
    free(self);
    exit(1);
  }

  // initializing each element as a Scalar
  for (int i = 0; i < self->size; i++) {
    if (data) {
      self->data[i] = *initialize_scalars(data[i], dtype, NULL, 0);
    } else {
      self->data[i] = *initialize_scalars(0.0f, dtype, NULL, 0);
    }
  }
  self->dtype = dtype;
  return self;
}

void delete_tensor(Tensor* tensor) {
  if (!tensor) return;
  for (int i = 0; i < tensor->size; i++) {
    cleanup(&tensor->data[i]);
  }
  free(tensor->data);
  free(tensor->strides);
  free(tensor->shape);
  free(tensor->aux);
  free(tensor);
}

void delete_shape(Tensor* tensor) {
  if (tensor->shape != NULL) {
    free(tensor->shape);
    tensor->shape = NULL;
  }
}

void delete_data(Tensor* tensor) {
  if (tensor->data != NULL) {
    free(tensor->data);
    free(tensor->aux);
    tensor->data = NULL;
    tensor->aux = NULL;
  }
}

void delete_strides(Tensor* tensor) {
  if (tensor->strides != NULL) {
    free(tensor->strides);
    tensor->strides = NULL;
  }
}

void delete_backstrides(Tensor* tensor) {
  if (tensor->backstrides != NULL) {
    free(tensor->backstrides);
    tensor->backstrides = NULL;
  }
}

Tensor* add_tensor(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for addition\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  add_tensor_cpu(a, b, out);
  return out;
}

Tensor* sub_tensor(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for subtraction\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  sub_tensor_cpu(a, b, out);
  return out;
}

Tensor* elemwise_mul_tensor(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  mul_tensor_cpu(a, b, out);
  return out;
}

Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  Tensor* out = create_tensor(NULL, broadcasted_shape, max_ndim, a->dtype);
  add_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  return out;
}

Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  Tensor* out = create_tensor(NULL, broadcasted_shape, max_ndim, a->dtype);
  sub_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  return out;
}

Tensor* elemwise_mul_broadcasted_tensor(Tensor* a, Tensor* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  Tensor* out = create_tensor(NULL, broadcasted_shape, max_ndim, a->dtype);
  mul_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  return out;
}

Tensor* matmul_tensor(Tensor* a, Tensor* b) {
  if (a->shape[1] != b->shape[0]) {
    fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    exit(1);
  }
  int ndim = a->ndim + b->ndim - 2;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  for (int i = 0; i < a->ndim - 1; i++) {
    shape[i] = a->shape[i];
  }
  for (int i = a->ndim - 1; i < ndim; i++) {
    shape[i] = a->shape[i - a->ndim + 2];
  }
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, shape, ndim, a->dtype);
  matmul_tensor_cpu(a, b, out);
  return out;
}

Tensor* batched_matmul_tensor(Tensor* a, Tensor* b) {
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Incompatible shapes for batched multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], a->shape[1]);
    exit(1);
  }
  if (a->shape[2] != b->shape[1]) {
    fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    exit(1);
  }
  int ndim = 3, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, shape, ndim, a->dtype);
  batched_matmul_tensor_cpu(a, b, out);
  return out;
}

Tensor* broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b) {
  if (a->shape[1] != b->shape[1]) {
    fprintf(stderr, "Incompatible shapes for broadcasted batched matrix multiplication %dx%d and %dx%dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1], b->shape[2]);
    exit(1);
  }
  int ndim = 3, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, shape, ndim, a->dtype);
  broadcasted_matmul_tensor_cpu(a, b, out, shape, size);
  return out;
}

Tensor* tensor_div_tensor(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  div_tensor_cpu(a, b, out);
  return out;
}

Tensor* scalar_mul_tensor(Tensor* a, Scalar* b) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  scalar_mul_tensor_cpu(a, b, out);
  return out;
}

Tensor* tensor_div_scalar(Tensor* a, Scalar* b) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  tensor_div_scalar_cpu(a, b, out);
  return out;
}

Tensor* scalar_div_tensor(Scalar* a, Tensor* b) {
  Tensor* out = create_tensor(NULL, b->shape, b->ndim, a->dtype);
  scalar_div_tensor_cpu(b, a, out);
  return out;
}

Tensor* tensor_pow_scalar(Tensor* a, Scalar* exp) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  tensor_pow_scalar_cpu(a, exp, out);
  return out;
}

Tensor* scalar_pow_tensor(Scalar* base, Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  scalar_pow_tensor_cpu(a, base, out);
  return out;
}

Tensor* log_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  log_tensor_cpu(a, out);
  return out;
}

Tensor* sum_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  sum_tensor_cpu(a, out, axis_size, shape, axis);
  if (keepdim) {
    if (axis == -1) {
      ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = 1;
      }
    } else {
      shape = (int*)malloc(a->ndim * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = a->shape[i];
      }
      shape[axis] = 1, ndim = a->ndim;
    }
  }
  return out;
}

Tensor* max_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  max_tensor_cpu(a, out, axis_size, shape, axis);
  if (keepdim) {
    if (axis == -1) {
      ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = 1;
      }
    } else {
      shape = (int*)malloc(a->ndim * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = a->shape[i];
      }
      shape[axis] = 1, ndim = a->ndim;
    }
  }
  return out;
}

Tensor* min_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  min_tensor_cpu(a, out, axis_size, shape, axis);
  if (keepdim) {
    if (axis == -1) {
      ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = 1;
      }
    } else {
      shape = (int*)malloc(a->ndim * sizeof(int));
      for (int i = 0; i < a->size; i++) {
        shape[i] = a->shape[i];
      }
      shape[axis] = 1, ndim = a->ndim;
    }
  }
  return out;
}

Tensor* sin_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  sin_tensor_cpu(a, out);
  return out;
}

Tensor* cos_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  cos_tensor_cpu(a, out);
  return out;
}

Tensor* gelu_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  gelu_tensor_cpu(a, out);
  return out;
}

Tensor* swiglu_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  swiglu_tensor_cpu(a, out);
  return out;
}

Tensor* silu_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  silu_tensor_cpu(a, out);
  return out;
}

Tensor* sigmoid_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  sigmoid_tensor_cpu(a, out);
  return out;
}

Tensor* tanh_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  tanh_tensor_cpu(a, out);
  return out;
}

Tensor* relu_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  relu_tensor_cpu(a, out);
  return out;
}

Tensor* reshape_tensor(Tensor* a, int* new_shape, int new_ndim) {
  int ndim = new_ndim, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    shape[i] = new_shape[i];
  }
  int size = 1;
  for (int i = 0; i < new_ndim; i++) {
    size *= shape[i];
  }
  if (size != a->size) {
    fprintf(stderr, "Can't reshape the tensor. tensor's size doesn't match the target size: %d != %d", a->size, size);
  }
  Tensor* out = create_tensor(NULL, shape, ndim, a->dtype);
  reassign_tensor_cpu(a, out);
  return out;
}

Tensor* transpose_tensor(Tensor* a) {
  int ndim = a->ndim, *shape = (int*)malloc(ndim * sizeof(int)), size = a->size;
    if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    shape[i] = a->shape[ndim - 1 - i];
  }
  Tensor* out = create_tensor(NULL, shape, ndim, a->dtype);
  switch(ndim) {
    case 1:
      transpose_1d_tensor_cpu(a, out);
      break;
    case 2:
      transpose_2d_tensor_cpu(a, out);
      break;
    case 3:
      transpose_3d_tensor_cpu(a, out);
      break;
    default:
      fprintf(stderr, "Transpose supported only for 3-dim tensor");
      exit(1);
  }
  return out;
}

void make_contiguous(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  make_contiguous_tensor_cpu(a, out);
}

Tensor* equal_tensor(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same dimensions %d and %d for equal", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  equal_tensor_cpu(a, b, out);
  return out;
}

Tensor* equal_broadcasted_tensor(Tensor* a, Tensor* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "Shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  Tensor* out = create_tensor(NULL, broadcasted_shape, max_ndim, a->dtype);
  equal_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  return out;
}

Tensor* zeros_like_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  ones_like_tensor_cpu(a->size, out);
  return out;
}

Tensor* ones_like_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->dtype);
  ones_like_tensor_cpu(a->size, out);
  return out;
}

// helper function to truncate elements in a single row
void truncate_row(const float* row, int length, int max_display, char* result) {
  strcat(result, "  [");
  if (length > max_display) {
    for (int i = 0; i < max_display / 2; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, buffer);
      strcat(result, ", ");
    }
    strcat(result, "...");
    for (int i = length - max_display / 2; i < length; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, ", ");
      strcat(result, buffer);
    }

    // removing trailing comma and space
    if (result[strlen(result) - 2] == ',') {
      result[strlen(result) - 2] = '\0';
    }
  } else {
    for (int i = 0; i < length; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, buffer);
      if (i != length - 1) strcat(result, ", ");
    }
  }
  strcat(result, "]");
}

// recursive function to format data as a nested array with truncation
void format_tensor(const float* data, const int* shape, int ndim, int level, char* result) {
  if (ndim == 1) {
    truncate_row(data, shape[0], 8, result);
    return;
  }

  strcat(result, "[\n");
  int rows_to_display = shape[0] > 4 ? 2 : shape[0]; // truncate rows if needed
  for (int i = 0; i < rows_to_display; i++) {
    if (i > 0) strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    format_tensor(data + i * shape[1], shape + 1, ndim - 1, level + 1, result);
  }

  if (shape[0] > 4) {
    strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    strcat(result, "...");
    strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    for (int i = shape[0] - 2; i < shape[0]; i++) {
      if (i > shape[0] - 2) strcat(result, ",\n");
      format_tensor(data + i * shape[1], shape + 1, ndim - 1, level + 1, result);
    }
  }
  strcat(result, "\n]");
}

void print_tensor(Tensor* a) {
  char result[4096] = "";
  format_tensor(a->aux, a->shape, a->ndim, 0, result);
  printf("tensor(%s, dtype=drop.%s)\n", result, dtype_to_string(a->dtype));
}