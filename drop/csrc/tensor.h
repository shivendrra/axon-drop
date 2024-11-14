/* 
  - tensor.h header file for tensor.cpp & Tensor
  - contains the wrapper over Scalar values & computes autograd for each value
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libtensor.so tensor.cpp dtype.cpp scalar.cpp
    -- '.dll': g++ -shared -o libtensor.dll tensor.cpp dtype.cpp scalar.cpp
*/

#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include "scalar.h"

// struct for a one-dimensional Tensor
typedef struct Tensor {
  Scalar* data;             // single-dimensional array of Scalars
  int* shape;               // array holding dimensions of the tensor
  int* strides;             // strides array to handle multi-dim indexing
  int ndim;                 // no of dim in tensor (1d structure)
  int size;                 // total elements in the tensor
  DType dtype;              // data-type of the tensor
  struct Tensor** _prev;    // track previous Tensors for autograd
  int _prev_size;           // no of previous Tensors
  // void (*_backward)(struct Tensor*);  // backward function for autograd
} Tensor;

// dynamic array struct to manage dynamic tensor storage for autograd
typedef struct DynamicTensorArray {
  Tensor** data;
  size_t size;
  size_t capacity;
} DynamicTensorArray;

extern "C" {
  Tensor* initialize_tensor(double* data, DType dtype, int* shape, int ndim);
  void calculate_strides(Tensor* t);        // calculate strides based on the tensor's shape
  int get_offset(int* index, Tensor* t);    // calculate flattened offset for a given multi-dimensional index
  void delete_tensor(Tensor* tensor);
  void delete_shape(Tensor* tensor);
  void delete_strides(Tensor* tensor);
  void delete_data(Tensor* tensor);

  Tensor* add_tensor(Tensor* a, Tensor* b);
  Tensor* mul_tensor(Tensor* a, Tensor* b);
  Tensor* sub_tensor(Tensor* a, Tensor* b);
  Tensor* neg_tensor(Tensor* a);
  Tensor* pow_tensor(Tensor* a, float exp);
  Tensor* relu_tensor(Tensor* a);
  Tensor* gelu_tensor(Tensor* a);
  Tensor* tanh_tensor(Tensor* a);
  Tensor* sigmoid_tensor(Tensor* a);
  Tensor* silu_tensor(Tensor* a);
  Tensor* swiglu_tensor(Tensor* a);

  void broadcast_recursive(Scalar* source, Scalar* target, int* source_shape, int* target_shape, int ndim, int source_dim, int target_index, int* strides);
  void broadcast_data(Scalar* data, int* shape, int ndim, Scalar** result, int* target_shape, int target_ndim);
  void backward_tensor(Tensor* t);

  double get_tensor_data(Tensor* t, int index);
  double get_tensor_grad(Tensor* t, int index);
  void set_tensor_data(Tensor* t, int index, double value);
  void set_tensor_grad(Tensor* t, int index, double value);
  void print_tensor(Tensor* t);
}

#endif