#ifndef SCALAR_H
#define SCALAR_H

#include <stdlib.h>
#include "dtype.h"

typedef struct Scalar {
  void* data;
  void* grad;
  DType dtype;
  struct Scalar** _prev;
  int _prev_size;
  void (*_backward)(struct Scalar*);
  double aux;
} Scalar;

typedef struct DynamicArray {
  Scalar** data;
  size_t size;
  size_t capacity;
} DynamicArray;

extern "C" {
  Scalar* initialize_scalars(double data, DType dtype, Scalar** child, int child_size);
  void noop_backward(Scalar* v);

  Scalar* add_val(Scalar* a, Scalar* b);
  void add_backward(Scalar* v);
  Scalar* mul_val(Scalar* a, Scalar* b);
  void mul_backward(Scalar* v);
  Scalar* pow_val(Scalar* a, float exp);
  void pow_backward(Scalar* v);

  Scalar* negate(Scalar* a);
  Scalar* sub_val(Scalar* a, Scalar* b);
  Scalar* div_val(Scalar* a, Scalar* b);

  Scalar* relu(Scalar* a);
  void relu_backward(Scalar* v);
  Scalar* sigmoid(Scalar* a);
  void sigmoid_backward(Scalar* v);
  Scalar* tan_h(Scalar* a);
  void tanh_backward(Scalar* v);
  Scalar* silu(Scalar* a);
  void silu_backward(Scalar* v);
  Scalar* gelu(Scalar* a);
  void gelu_backward(Scalar* v);
  Scalar* swiglu(Scalar* a);
  void swiglu_backward(Scalar* v);

  void build_topo(Scalar* v, DynamicArray* topo, DynamicArray* visited);
  void backward(Scalar* v);
  void print(Scalar* v);
  void cleanup(Scalar* v);
  double get_scalar_data(Scalar* v);
  double get_scalar_grad(Scalar* v);
  void set_scalar_data(Scalar* v, double value);
  void set_scalar_grad(Scalar* v, double value);
}

#endif