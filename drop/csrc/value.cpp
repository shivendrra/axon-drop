#include "value.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

Value* initialize_value(double* data) {
  Value* v = (Value*)malloc(sizeof(Value));
  v->data = (double*)malloc(sizeof(double));
  if (data) {
    memcpy(v->data, data, sizeof(double));
  } else {
    *(v->data) = 0.0;
  }
  v->grad = (double*)malloc(sizeof(double));
  *(v->grad) = 0.0;
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

  *(a->grad) += *(v->grad);
  *(b->grad) += *(v->grad);
}

Value* add_val(Value* a, Value* b) {
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = initialize_value(NULL);
  *(out->data) = *(a->data) + *(b->data);
  out->_prev = children;
  *(out->_prev_size) = 2;
  out->_backward = add_backward;
  return out;
}

void mul_backward(Value* v) {
  Value* a = v->_prev[0];
  Value* b = v->_prev[1];

  *(a->grad) += *(b->data) * *(v->grad);
  *(b->grad) += *(a->data) * *(v->grad);
}

Value* mul_val(Value* a, Value* b) {
  Value **children = (Value**)malloc(2 * sizeof(Value*));
  children[0] = a;
  children[1] = b;

  Value* out = initialize_value(NULL);
  *(out->data) = *(a->data) * *(b->data);
  out->_prev = children;
  *(out->_prev_size) = 2;
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Value* v) {
  Value* a = v->_prev[0];
  *(a->grad) += *(v->exp) * std::pow(*(a->data), *(v->exp) - 1) * *(v->grad);
}

Value* pow_val(Value* a, double* exp) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL);
  *(out->data) = std::pow(*(a->data), *exp);
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = pow_backward;
  *(out->exp) = *exp;
  return out;
}

Value* negate(Value* a) {
  double neg_one = -1.0;
  Value* neg_one_val = initialize_value(&neg_one);
  return mul_val(a, neg_one_val);
}

Value* sub_val(Value* a, Value* b) {
  return add_val(a, negate(b));
}

void relu_backward(Value* v) {
  Value* a = v->_prev[0];
  *(a->grad) += (*(v->data) > 0) * *(v->grad);
}

Value* relu(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL);
  *(out->data) = *(a->data) > 0 ? *(a->data) : 0;
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = relu_backward;
  return out;
}

void sigmoid_backward(Value* v) {
  Value* a = v->_prev[0];
  double sig = 1 / (1 + std::exp(-(*(v->data))));
  *(a->grad) += sig * (1 - sig) * *(v->grad);
}

Value* sigmoid(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL);
  *(out->data) = 1 / (1 + std::exp(-(*(a->data))));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = sigmoid_backward;
  return out;
}

void tanh_backward(Value* v) {
  Value* a = v->_prev[0];
  *(a->grad) += (1 - std::pow(*(v->data), 2)) * *(v->grad);
}

Value* tan_h(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL);
  *(out->data) = std::tanh(*(a->data));
  out->_prev = children;
  *(out->_prev_size) = 1;
  out->_backward = tanh_backward;
  return out;
}

void silu_backward(Value* v) {
  Value* a = v->_prev[0];
  double sig = 1 / (1 + std::exp(-(*(a->data))));
  *(a->grad) += (*(v->grad)) * (sig + (*(v->data)) * (1 - sig));
}

Value* silu(Value* a) {
  Value **children = (Value**)malloc(1 * sizeof(Value*));
  children[0] = a;

  Value* out = initialize_value(NULL);
  *(out->data) = *(a->data) / (1 + std::exp(-(*(a->data))));
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
  *(v->grad) = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward(*it);
  }
}

double get_data(const Value* v) {
  return *(v->data);
}

double get_grad(const Value* v) {
  return *(v->grad);
}