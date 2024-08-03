#include "value.h"
#include "dtype.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

Value* initialize_value(double* data, DType dtype) {
  Value* v = (Value*)malloc(sizeof(Value));
  v->data = initialize_data(data ? *data : 0.0, dtype);
  v->grad = initialize_data(0.0, dtype);
  v->dtype = dtype;
  v->exp = (double*)malloc(sizeof(double));
  *(v->exp) = 0.0;
  v->_prev = NULL;
  v->_prev_size = (int*)malloc(sizeof(int));
  *(v->_prev_size) = 0;
  v->_backward = noop_backward;
  return v;
}

void noop_backward(Value* v) { }

void add_backward(Value* v) {
  Value* a = v->_prev[0];
  Value* b = v->_prev[1];
  if (a->dtype != b->dtype || a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in add_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_grad = get_data_as_double(a->grad, a->dtype);
  double b_grad = get_data_as_double(b->grad, b->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + grad_val);
  set_data_from_double(b->grad, b->dtype, b_grad + grad_val);
}

Value* add_val(Value* a, Value* b) {
  if (a->dtype != b->dtype) {
    throw std::invalid_argument("Data types do not match in add_val");
  }
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double b_data = get_data_as_double(b->data, b->dtype);
  set_data_from_double(out->data, out->dtype, a_data + b_data);
  out->_prev = children;
  *(out->_prev_size) = 2;
  out->_backward = add_backward;
  return out;
}

void mul_backward(Value* v) {
  Value* a = v->_prev[0];
  Value* b = v->_prev[1];
  if (a->dtype != b->dtype || a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in mul_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double b_data = get_data_as_double(b->data, b->dtype);
  double a_grad = get_data_as_double(a->grad, a->dtype);
  double b_grad = get_data_as_double(b->grad, b->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + b_data * grad_val);
  set_data_from_double(b->grad, b->dtype, b_grad + a_data * grad_val);
}

Value* mul_val(Value* a, Value* b) {
  if (a->dtype != b->dtype) {
    throw std::invalid_argument("Data types do not match in mul_val");
  }
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double b_data = get_data_as_double(b->data, b->dtype);
  set_data_from_double(out->data, out->dtype, a_data * b_data);
  out->_prev = children;
  *(out->_prev_size) = 2;
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Value* v) {
  Value* a = v->_prev[0];
  if (a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in pow_backward");
  }
  double exp_val = *(v->exp);
  double a_data = get_data_as_double(a->data, a->dtype);
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_grad = get_data_as_double(a->grad, a->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + exp_val * std::pow(a_data, exp_val - 1) * grad_val);
}

Value* pow_val(Value* a, double* exp) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  set_data_from_double(out->data, out->dtype, std::pow(a_data, *exp));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = pow_backward;
  *(out->exp) = *exp;
  return out;
}

Value* negate(Value* a) {
  double neg_one = -1.0;
  Value* neg_one_val = initialize_value(&neg_one, a->dtype);
  return mul_val(a, neg_one_val);
}

Value* sub_val(Value* a, Value* b) {
  return add_val(a, negate(b));
}

void relu_backward(Value* v) {
  Value* a = v->_prev[0];
  if (a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in relu_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double a_grad = get_data_as_double(a->grad, a->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + (a_data > 0 ? grad_val : 0));
}

Value* relu(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  set_data_from_double(out->data, out->dtype, a_data > 0 ? a_data : 0);
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = relu_backward;
  return out;
}

void sigmoid_backward(Value* v) {
  Value* a = v->_prev[0];
  if (a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in sigmoid_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double sig = 1 / (1 + std::exp(-a_data));
  double a_grad = get_data_as_double(a->grad, a->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + sig * (1 - sig) * grad_val);
}

Value* sigmoid(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  set_data_from_double(out->data, out->dtype, 1 / (1 + std::exp(-a_data)));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = sigmoid_backward;
  return out;
}

void tanh_backward(Value* v) {
  Value* a = v->_prev[0];
  if (a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in tanh_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double a_grad = get_data_as_double(a->grad, a->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + (1 - std::pow(a_data, 2)) * grad_val);
}

Value* tan_h(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  set_data_from_double(out->data, out->dtype, std::tanh(a_data));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = tanh_backward;
  return out;
}

void silu_backward(Value* v) {
  Value* a = v->_prev[0];
  if (a->dtype != v->dtype) {
    throw std::invalid_argument("Data types do not match in silu_backward");
  }
  double grad_val = get_data_as_double(v->grad, v->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  double sig = 1 / (1 + std::exp(-a_data));
  double a_grad = get_data_as_double(a->grad, a->dtype);
  set_data_from_double(a->grad, a->dtype, a_grad + grad_val * (sig + a_data * (1 - sig)));
}

Value* silu(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL, a->dtype);
  double a_data = get_data_as_double(a->data, a->dtype);
  set_data_from_double(out->data, out->dtype, a_data / (1 + std::exp(-a_data)));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = silu_backward;
  return out;
}

void build_topo(Value* v, std::vector<Value*>& topo, std::vector<Value*>& visited) {
  for (auto vi : visited) {
    if (vi == v) return;
  }
  visited.push_back(v);
  for (int i = 0; i < *(v->_prev_size); ++i) {
    build_topo(v->_prev[i], topo, visited);
  }
  topo.push_back(v);
}

void backward(Value* v) {
  std::vector<Value*> topo;
  std::vector<Value*> visited;
  build_topo(v, topo, visited);
  set_data_from_double(v->grad, v->dtype, 1.0);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward(*it);
  }
}

double get_data_as_double(const Value* v) {
  return get_data_as_double(v->data, v->dtype);
}

double get_grad_as_double(const Value* v) {
  return get_data_as_double(v->grad, v->dtype);
}

void set_data_from_double(Value* v, double value) {
  set_data_from_double(v->data, v->dtype, value);
}

void set_grad_from_double(Value* v, double value) {
  set_data_from_double(v->grad, v->dtype, value);
}