#include "tensor.h"
#include <iostream>
#include <cmath>

Tensor* initialize_tensor(size_t* shape, size_t ndim, DType dtype, double init_value) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
  tensor->ndim = ndim;
  tensor->dtype = dtype;

  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; i++) {
    tensor->shape[i] = shape[i];
    total_elements *= shape[i];
  }

  tensor->data = (Scalar**)malloc(total_elements * sizeof(Scalar*));
  for (size_t i = 0; i < total_elements; i++) {
    tensor->data[i] = initialize_scalars(init_value, dtype, nullptr, 0);
  }

  return tensor;
}

void tensor_cleanup(Tensor* tensor) {
  size_t total_elements = 1;
  for (size_t i = 0; i < tensor->ndim; i++) {
    total_elements *= tensor->shape[i];
  }

  for (size_t i = 0; i < total_elements; i++) {
    cleanup(tensor->data[i]);
  }

  free(tensor->data);
  free(tensor->shape);
  free(tensor);
}

size_t tensor_index_to_offset(size_t* indices, size_t* shape, size_t ndim) {
  size_t offset = 0;
  size_t stride = 1;
  for (size_t i = 0; i < ndim; i++) {
    offset += indices[ndim - i - 1] * stride;
    stride *= shape[ndim - i - 1];
  }
  return offset;
}

Scalar* tensor_get_value(Tensor* tensor, size_t* indices) {
  size_t offset = tensor_index_to_offset(indices, tensor->shape, tensor->ndim);
  return tensor->data[offset];
}

void tensor_set_value(Tensor* tensor, size_t* indices, double value) {
  size_t offset = tensor_index_to_offset(indices, tensor->shape, tensor->ndim);
  set_scalar_data(tensor->data[offset], value);
}

Tensor* tensor_relu(Tensor* tensor) {
  size_t total_elements = 1;
  for (size_t i = 0; i < tensor->ndim; i++) {
    total_elements *= tensor->shape[i];
  }

  Tensor* result = (Tensor*)malloc(sizeof(Tensor));
  result->shape = (size_t*)malloc(tensor->ndim * sizeof(size_t));
  result->ndim = tensor->ndim;
  result->dtype = tensor->dtype;

  result->data = (Scalar**)malloc(total_elements * sizeof(Scalar*));
  for (size_t i = 0; i < total_elements; i++) {
    result->data[i] = relu(tensor->data[i]);
  }

  return result;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
  size_t total_elements = 1;
  for (size_t i = 0; i < a->ndim; i++) {
    total_elements *= a->shape[i];
  }

  Tensor* result = (Tensor*)malloc(sizeof(Tensor));
  result->shape = (size_t*)malloc(a->ndim * sizeof(size_t));
  result->ndim = a->ndim;
  result->dtype = a->dtype;

  result->data = (Scalar**)malloc(total_elements * sizeof(Scalar*));
  for (size_t i = 0; i < total_elements; i++) {
    result->data[i] = add_val(a->data[i], b->data[i]);
  }

  return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
  size_t total_elements = 1;
  for (size_t i = 0; i < a->ndim; i++) {
    total_elements *= a->shape[i];
  }

  Tensor* result = (Tensor*)malloc(sizeof(Tensor));
  result->shape = (size_t*)malloc(a->ndim * sizeof(size_t));
  result->ndim = a->ndim;
  result->dtype = a->dtype;

  result->data = (Scalar**)malloc(total_elements * sizeof(Scalar*));
  for (size_t i = 0; i < total_elements; i++) {
    result->data[i] = mul_val(a->data[i], b->data[i]);
  }

  return result;
}

void print_tensor(Tensor* tensor) {
  std::cout << "Tensor shape: [";
  for (size_t i = 0; i < tensor->ndim; i++) {
    std::cout << tensor->shape[i];
    if (i < tensor->ndim - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  size_t total_elements = 1;
  for (size_t i = 0; i < tensor->ndim; i++) {
    total_elements *= tensor->shape[i];
  }

  std::cout << "Tensor data:" << std::endl;
  for (size_t i = 0; i < total_elements; i++) {
    print(tensor->data[i]);
    if ((i + 1) % tensor->shape[tensor->ndim - 1] == 0) {
      std::cout << std::endl;
    } else {
      std::cout << " ";
    }
  }
}
