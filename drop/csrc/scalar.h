#ifndef Scalar_H
#define Scalar_H

#include "dtype.h"
#include <vector>

typedef struct Scalar {
  void* data;
  void* grad;
  DType dtype;
  struct Scalar** _prev;
  int* _prev_size;
  void (*_backward)(struct Scalar*);
  double* exp;
} Scalar;

extern "C" {
  Scalar* initialize_Scalar(double* data, DType dtype);
  void noop_backward(Scalar* v);

  Scalar* add_val(Scalar* a, Scalar* b);
  void add_backward(Scalar* v);
  Scalar* mul_val(Scalar* a, Scalar* b);
  void mul_backward(Scalar* v);
  Scalar* pow_val(Scalar* a, double* exp);
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

  void build_topo(Scalar* v, std::vector<Scalar*>& topo, std::vector<Scalar*>& visited);
  void backward(Scalar* v);

  double get_data_as_double(const Scalar* v);
  double get_grad_as_double(const Scalar* v);
  void set_data_from_double(Scalar* v, double Scalar);
}

#endif