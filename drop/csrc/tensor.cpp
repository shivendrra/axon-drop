#include "tensor.h"
#include "scalar.h"
#include "dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <algorithm>

const int MAX_ITEMS = 8; // Threshold for truncation

void calculate_strides(Tensor* t) {
  t->strides = (int*)malloc(t->ndim * sizeof(int));
  t->strides[t->ndim - 1] = 1;
  for (int i = t->ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
  }
}

int get_offset(int* index, Tensor* t) {
  int offset = 0;
  for (int i = 0; i < t->ndim; i++) {
    offset += index[i] * t->strides[i];
  }
  return offset;
}

Tensor* initialize_tensor(double *input_data, DType dtype, int *shape, int ndim) {
  Tensor *self = (Tensor*)malloc(sizeof(Tensor));
  self->dtype = dtype;
  self->ndim = ndim;
  self->shape = (int*)malloc(ndim * sizeof(int));
  self->size = 1;

  for (int i=0; i<ndim; i++) {
    self->shape[i] = shape[i];
    self->size *= shape[i];
  }

  self->data = (Scalar*)malloc(self->size * sizeof(Scalar));
  for (int i=0; i<self->size; i++) {
    self->data[i] = *initialize_scalars(input_data[i], dtype, NULL, 0);
  }

  self->_prev = NULL;
  self->_prev_size = 0;

  calculate_strides(self);

  return self;
}

void delete_tensor(Tensor* tensor) {
  if (tensor != NULL) {
    free(tensor);
    tensor = NULL;
  }
}

void delete_shape(Tensor* tensor) {
  if (tensor->shape != NULL) {
    free(tensor->shape);
    tensor->shape = NULL;
  }
}

void delete_strides(Tensor* tensor) {
  if (tensor->strides != NULL) {
    free(tensor->strides);
    tensor->strides = NULL;
  }
}

void delete_data(Tensor* tensor) {
  if (tensor->data != NULL) {
    free(tensor->data);
  }
  tensor->data = NULL;
}

double get_tensor_data(Tensor* t, int index) {
  return get_scalar_data(&t->data[index]);
}

double get_tensor_grad(Tensor* t, int index) {
  return get_scalar_grad(&t->data[index]);
}

void set_tensor_data(Tensor* t, int index, double value) {
  set_scalar_data(&t->data[index], value);
}

void set_tensor_grad(Tensor* t, int index, double value) {
  set_scalar_grad(&t->data[index], value);
}

int calculate_size(int* shape, int ndim) {
  int size = 1;
  for (int i = 0; i < ndim; ++i) {
    size *= shape[i];
  }
  return size;
}

void broadcast_shape(const Tensor* a, const Tensor* b, int** target_shape, int* target_ndim) {
  int max_dim = std::max(a->ndim, b->ndim);
  *target_shape = (int*)malloc(max_dim * sizeof(int));
  
  for (int i = 0; i < max_dim; ++i) {
    int dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
    int dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
    
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::invalid_argument("Incompatible shapes for broadcasting.");
    }
    
    (*target_shape)[max_dim - 1 - i] = std::max(dim_a, dim_b);
  }
  *target_ndim = max_dim;
}

void broadcast_recursive(Scalar* source, Scalar* target, int* source_shape, int* target_shape, int ndim, int source_dim, int target_index, int* strides) {
  if (source_dim == ndim) {
    target[target_index] = source[0];
    return;
  }

  int source_size = source_shape[source_dim];
  int target_size = target_shape[source_dim];
  
  if (source_size == 1) {
    for (int i = 0; i < target_size; ++i) {
      broadcast_recursive(source, target, source_shape, target_shape, ndim, source_dim + 1, target_index + i * strides[source_dim], strides);
    }
  } else {
    for (int i = 0; i < target_size; ++i) {
      broadcast_recursive(&source[i * strides[source_dim]], target, source_shape, target_shape, ndim, source_dim + 1, target_index + i * strides[source_dim], strides);
    }
  }
}

void broadcast_data(Scalar* data, int* shape, int ndim, Scalar** result, int* target_shape, int target_ndim) {
  int target_size = calculate_size(target_shape, target_ndim);
  *result = (Scalar*)malloc(target_size * sizeof(Scalar));

  int* strides = (int*)malloc(ndim * sizeof(int));
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  broadcast_recursive(data, *result, shape, target_shape, ndim, 0, 0, strides);
  free(strides);
}

Tensor* add_tensor(Tensor* a, Tensor* b) {
  int* target_shape;
  int target_ndim;
  broadcast_shape(a, b, &target_shape, &target_ndim);
  
  Scalar* data_a;
  Scalar* data_b;
  broadcast_data(a->data, a->shape, a->ndim, &data_a, target_shape, target_ndim);
  broadcast_data(b->data, b->shape, b->ndim, &data_b, target_shape, target_ndim);
  
  Tensor* out = initialize_tensor(NULL, a->dtype, target_shape, target_ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *add_val(&data_a[i], &data_b[i]);
  }
  out->_prev = (Tensor**)malloc(2 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev[1] = b;
  out->_prev_size = 2;

  free(data_a);
  free(data_b);
  free(target_shape);
  return out;
}

Tensor* mul_tensor(Tensor* a, Tensor* b) {
  int* target_shape;
  int target_ndim;
  broadcast_shape(a, b, &target_shape, &target_ndim);

  Scalar* data_a;
  Scalar* data_b;
  broadcast_data(a->data, a->shape, a->ndim, &data_a, target_shape, target_ndim);
  broadcast_data(b->data, b->shape, b->ndim, &data_b, target_shape, target_ndim);

  Tensor* out = initialize_tensor(NULL, a->dtype, target_shape, target_ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *mul_val(&data_a[i], &data_b[i]);
  }
  out->_prev = (Tensor**)malloc(2 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev[1] = b;
  out->_prev_size = 2;

  free(data_a);
  free(data_b);
  free(target_shape);
  return out;
}

Tensor* sub_tensor(Tensor* a, Tensor* b) {
  int* target_shape;
  int target_ndim;
  broadcast_shape(a, b, &target_shape, &target_ndim);

  Scalar* data_a;
  Scalar* data_b;
  broadcast_data(a->data, a->shape, a->ndim, &data_a, target_shape, target_ndim);
  broadcast_data(b->data, b->shape, b->ndim, &data_b, target_shape, target_ndim);

  Tensor* out = initialize_tensor(NULL, a->dtype, target_shape, target_ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *sub_val(&data_a[i], &data_b[i]);
  }
  out->_prev = (Tensor**)malloc(2 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev[1] = b;
  out->_prev_size = 2;

  free(data_a);
  free(data_b);
  free(target_shape);
  return out;
}

// power void
Tensor* pow_tensor(Tensor* a, float exp) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *pow_val(&data_a[i], exp);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* neg_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *negate(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* relu_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *relu(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* gelu_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *gelu(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* sigmoid_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *sigmoid(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* tanh_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *tan_h(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* swiglu_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *swiglu(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

Tensor* silu_tensor(Tensor* a) {
  Scalar* data_a;
  Tensor* out = initialize_tensor(NULL, a->dtype, a->shape, a->ndim);
  for (int i = 0; i < out->size; ++i) {
    out->data[i] = *silu(&data_a[i]);
  }
  out->_prev = (Tensor**)malloc(1 * sizeof(Tensor*));
  out->_prev[0] = a;
  out->_prev_size = 1;

  free(data_a);
  return out;
}

void backward_tensor(Tensor* t) {
  if (t->size == 1) {   // check if tensor has only one element (scalar)
    backward(&t->data[0]);
  } else {
    fprintf(stderr, "Error: Backward can only be called on scalar values.\n");
  }
}

std::string format_element(double element, DType dtype) {
  std::ostringstream out;
  if (dtype == DType::INT8 || dtype == DType::INT16 || dtype == DType::INT32 || dtype == DType::INT64) {
    out << std::fixed << std::setprecision(0) << element; // Integer
  } else if (dtype == DType::FLOAT32) {
    out << std::fixed << std::setprecision(3) << element; // Float32
  } else if (dtype == DType::FLOAT64) {
    out << std::fixed << std::setprecision(4) << element; // Float64
  } else {
    out << element; // default
  }
  return out.str();
}

// function to format & print data recursively
void print_tensor_recursive(void* data, const int* shape, int ndim, int current_dim, int& index, int level, DType dtype) {
  if (current_dim == ndim - 1) {
    std::cout << "[";
    for (int i = 0; i < shape[current_dim]; ++i) {
      double value = get_data_as_double(static_cast<char*>(data) + index * dtype_size(dtype), dtype);
      std::cout << format_element(value, dtype);
      index++;
      if (i < shape[current_dim] - 1) {
        std::cout << ", ";
      }
      if (i == MAX_ITEMS / 2 - 1 && shape[current_dim] > MAX_ITEMS) {
        std::cout << ", ...";
        index += shape[current_dim] - MAX_ITEMS; // skip to the last part of the array
        i = shape[current_dim] - MAX_ITEMS / 2 - 1; // adjust i for the end truncation
      }
    }
    std::cout << "]";
  } else {
    std::cout << "[";
    for (int i = 0; i < shape[current_dim]; ++i) {
      if (i == MAX_ITEMS / 2 && shape[current_dim] > MAX_ITEMS) {
        std::cout << "  ...\n";
        i = shape[current_dim] - MAX_ITEMS / 2 - 1; // skip to the last part of the tensor
      } else {
        if (i > 0) std::cout << ",\n" << std::string((level + 1) * 2, ' '); // indentation
        print_tensor_recursive(data, shape, ndim, current_dim + 1, index, level + 1, dtype);
      }
    }
    std::cout << "]";
  }
}

void print_tensor(Tensor* t) {
  if (t == nullptr) {
    std::cerr << "Error: Tensor is null." << std::endl;
    return;
  }

  int index = 0;
  std::cout << "tensor(";
  print_tensor_recursive(t->data, t->shape, t->ndim, 0, index, 0, t->dtype);
  std::cout << ", dtype=" << dtype_to_string(t->dtype) << ")\n";
}