#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "tensor.h"
#include "scalar.h"
#include "cpu.h"
#include "dtype.h"

Tensor* create_tensor(float* data, int* shape, int ndim, char* device, DType dtype) {
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
  if (data != NULL) {
    // ff data is provided, initialize Scalars with values from data
    for (int i = 0; i < self->size; i++) {
      self->data[i] = *initialize_scalars(data[i], dtype, NULL, 0);
    }
  } else {
    // if data is NULL, initialize Scalars with default values (0.0)
    for (int i = 0; i < self->size; i++) {
      self->data[i] = *initialize_scalars(0.0f, dtype, NULL, 0);
    }
  }
  self->dtype = dtype;
  self->device = (char*)malloc(strlen(device) + 1);
  self->aux = (float*)malloc(self->size * sizeof(float));
  if (!self->aux) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
  for (int i = 0; i < self->size; i++) {
    self->aux[i] = get_data_as_float(self->data[i]->data, self->dtype);
  }
  if (device != NULL) {
    strcpy(self->device, device);
  } else {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

  return self;
}

void to_device(Tensor* tensor, char* device) {
  int device_id = 0;
  char *end_ptr, *device_type;
  long num = strtol(device, &end_ptr, 10);
  if (*end_ptr == '\0') {
    deivce_id = (int)num;
    device_type = new char[strlen("cuda") + 1];
    strcpy(device_type, "cuda");
  } else {
    device_type = new char[strlen("cpu") + 1];
    strcpy(device_type, "cpu");
  }
  if((strcmp(device_type, "cuda") == 0) && (strcmp(a->device, "cpu") == 0)) {
    cpu_to_cuda(a, device_id);
  } else if ((strcmp(device_type, "cpu") == 0) && (strcmp(a->device, "cuda") == 0)) {
    cuda_to_cpu(a);
  }
  free(device_type);
}

void delete_tensor(Tensor* tensor) {
  if (!tensor) return;
  for (int i = 0; i < tensor->size; i++) {
    cleanup(&tensor->data[i]);
  }
  free(tensor->data);
  free(tensor->strides);
  free(tensor->shape);
  free(tensor->device);
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
    tensor->data = NULL;
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

void delete_device(Tensor* tensor) {
  if (tensor->device != NULL) {
    free(tensor->device);
    tensor->device = NULL;
  }
}

void delete_aux(Tensor* tensor) {
  if (tensor->aux != NULL) {
    free(tensor->aux);
    tensor->aux = NULL;
  }
}

Tensor* add_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for addition\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    add_tensor_cpu(a, b, out);
  } else {
    add_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* sub_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for subtraction\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    sub_tensor_cpu(a, b, out);
  } else {
    sub_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* elemwise_mul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    mul_tensor_cpu(a, b, out);
  } else {
    mul_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    add_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  } else {
    add_broadcasted_tensor_cuda(a, b, out, broadcasted_shape, broadcasted_size);
  }
  return out;
}

Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    sub_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  } else {
    sub_broadcasted_tensor_cuda(a, b, out, broadcasted_shape, broadcasted_size);
  }
  return out;
}

Tensor* elemwise_mul_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    mul_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  } else {
    mul_broadcasted_tensor_cuda(a, b, out, broadcasted_shape, broadcasted_size);
  }
  return out;
}

Tensor* matmul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    matmul_tensor_cpu(a, b, out);
  } else {
    matmul_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* batched_matmul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    batched_matmul_tensor_cpu(a, b, out);
  } else {
    batched_matmul_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    broadcasted_matmul_tensor_cpu(a, b, out);
  } else {
    broadcasted_matmul_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* tensor_div_tensor(Tensor* a, Tensor* b) {
    if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    div_tensor_cpu(a, b, out);
  } else {
    div_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* scalar_mul_tensor(Tensor* a, float b) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    scalar_mul_tensor_cpu(a, b, out);
  } else {
    scalar_mul_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* tensor_div_scalar(Tensor* a, float b) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    tensor_div_scalar_cpu(a, b, out);
  } else {
    tensor_div_scalar_cuda(a, b, out);
  }
  return out;
}

Tensor* scalar_div_tensor(float a, Tensor* b) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    scalar_div_tensor_cpu(a, b, out);
  } else {
    scalar_div_tensor_cuda(a, b, out);
  }
  return out;
}

Tensor* tensor_pow_scalar(Tensor* a, float exp) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    tensor_pow_scalar_cpu(a, exp, out);
  } else {
    tensor_pow_scalar_cuda(a, exp, out);
  }
  return out;
}

Tensor* scalar_pow_tensor(float base, Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    scalar_pow_tensor_cpu(base, a, out);
  } else {
    scalar_pow_tensor_cuda(base, a, out);
  }
  return out;
}

Tensor* log_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    log_tensor_cpu(a, out);
  } else {
    log_tensor_cuda(a, out);
  }
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
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
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
    return create_tensor(out, a->shape, a->ndim, a->device, a->dtype);
  } else {
    sum_tensor_cuda(a, out, axis_size, shape, axis);
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
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
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
  } else {
    max_tensor_cuda(a, out, axis_size, shape, axis);
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
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
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
  } else {
  min_tensor_cuda(a, out, axis_size, shape, axis);
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
  }
  return out;
}

Tensor* sin_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    sin_tensor_cpu(a, out);
  } else {
    sin_tensor_cuda(a, out);
  }
  return out;
}

Tensor* cos_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    cos_tensor_cpu(a, out);
  } else {
    cos_tensor_cuda(a, out);
  }
  return out;
}

Tensor* sigmoid_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    sigmoid_tensor_cpu(a, out);
  } else {
    sigmoid_tensor_cuda(a, out);
  }
  return out;
}

Tensor* tanh_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    tanh_tensor_cpu(a, out);
  } else {
    tanh_tensor_cuda(a, out);
  }
  return out;
}

Tensor* relu_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    relu_tensor_cpu(a, out);
  } else {
    relu_tensor_cuda(a, out);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    reassign_tensor_cpu(a, out);
  } else {
    reassign_tensor_cuda(a, out);
  }
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
  Tensor* out = create_tensor(NULL, shape, ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
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
  } else {
    switch(ndim) {
      case 1:
        transpose_1d_tensor_cuda(a, out);
        break;
      case 2:
        transpose_2d_tensor_cuda(a, out);
        break;
      case 3:
        transpose_3d_tensor_cuda(a, out);
        break;
      default:
        fprintf(stderr, "Transpose supported only for 3-dim tensor");
        exit(1);
    }
  }
  return out;
}

void make_contiguous(Tensor* a) {
  int* new_strides = (int*)malloc(a->ndim * sizeof(int));
  if (new_strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
  }
  int stride = 1;
  for (int i = a->ndim - 1; i >= 0; i--) {
    new_strides[i] = stride;
    stride *= a->shape[i];
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    make_contagious_tensor_cpu(a, out, new_strides);
  } else {
    make_contagious_tensor_cuda(a, out, new_strides);
  }
}

Tensor* equal_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same dimensions %d and %d for equal", a->ndim, b->ndim);
    exit(1);
  }
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    equal_tensor_cpu(a, b, out);
  } else {
    equal_tensor_cuda(a, b, out);
  }
}

Tensor* equal_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
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
  Tensor* out = create_tensor(NULL, brodacasted_shape, max_ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    equal_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
  } else {
    equal_broadcasted_tensor_cuda(a, b, out, broadcasted_shape, broadcasted_size);
  }
  return out;
}

Tensor* zeros_like_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    ones_like_tensor_cpu(a, out);
  } else {
    ones_like_tensor_cuda(a, out);
  }
  return out;
}

Tensor* ones_like_tensor(Tensor* a) {
  Tensor* out = create_tensor(NULL, a->shape, a->ndim, a->device, a->dtype);
  if (strcmp(a->device, "cpu") == 0) {
    ones_like_tensor_cpu(a, out);
  } else {
    ones_like_tensor_cuda(a, out);
  }
  return out;
}