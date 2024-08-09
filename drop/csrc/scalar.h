#ifndef Scalar_H
#define Scalar_H

#include <stdlib.h>

typedef struct Scalar {
  double data;
  double grad;
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
  Scalar* initialize_scalars(double data, Scalar** child, int child_size);
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

  void build_topo(Scalar* v, DynamicArray* topo, DynamicArray* visited);
  void backward(Scalar* v);
  void print(Scalar* v);
  void cleanup(Scalar* v);
  double get_scalar_data(Scalar* v);
  double get_scalar_grad(Scalar* v);
}

#endif