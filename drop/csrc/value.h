#ifndef VALUE_H
#define VALUE_H

#include "dtype.h"
#include <vector>

typedef struct Value {
  void* data;
  void* grad;
  DType dtype;
  struct Value** _prev;
  int* _prev_size;
  void (*_backward)(struct Value*);
  double* exp;
} Value;

extern "C" {
  Value* initialize_value(double* data, DType dtype);
  void noop_backward(Value* v);

  Value* add_val(Value* a, Value* b);
  void add_backward(Value* v);
  Value* mul_val(Value* a, Value* b);
  void mul_backward(Value* v);
  Value* pow_val(Value* a, double* exp);
  void pow_backward(Value* v);
  
  Value* negate(Value* a);
  Value* sub_val(Value* a, Value* b);
  Value* div_val(Value* a, Value* b);
  
  Value* relu(Value* a);
  void relu_backward(Value* v);
  Value* sigmoid(Value* a);
  void sigmoid_backward(Value* v);
  Value* tan_h(Value* a);
  void tanh_backward(Value* v);
  Value* silu(Value* a);
  void silu_backward(Value* v);
  Value* gelu(Value* a);
  void gelu_backward(Value* v);
  Value* swiglu(Value* a);
  void swiglu_backward(Value* v);

  void build_topo(Value* v, std::vector<Value*>& topo, std::vector<Value*>& visited);
  void backward(Value* v);

  double get_data_as_double(const Value* v);
  double get_grad_as_double(const Value* v);
  void set_data_from_double(Value* v, double value);
}

#endif