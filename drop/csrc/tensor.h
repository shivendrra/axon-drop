#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include "scalar.h"

typedef struct Tensor {
  Scalar* data;
  size_t* shape;
  size_t ndim;
  size_t size;
} Tensor;

extern "C" {
  Tensor* initialize_tensor(size_t* shape, size_t ndim, DType dtype, double init_value);
  void tensor_set_value(Tensor* tensor, size_t* indices, double value);
  Scalar* tensor_get_value(Tensor* tensor, size_t* indices);
  void tensor_cleanup(Tensor* tensor);
  void print_tensor(Tensor* tensor);

  Tensor* tensor_add(Tensor* a, Tensor* b);
  Tensor* tensor_mul(Tensor* a, Tensor* b);
  Tensor* tensor_relu(Tensor* a);
  Tensor* tensor_sigmoid(Tensor* a);
  Tensor* tensor_tanh(Tensor* a);
}

// size_t calculate_tensor_size(const size_t* shape, size_t ndim);
// size_t flatten_index(const size_t* shape, size_t ndim, const size_t* indices);

#endif