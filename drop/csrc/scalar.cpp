#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "scalar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void noop_backward(Scalar *self) {}

Scalar* initialize_scalars(double data, Scalar** child, int child_size) {
  Scalar* self = (Scalar*)malloc(sizeof(Scalar));
  self->data = data;
  self->grad = 0.0;
  self->_prev_size = child_size;
  if (child_size > 0) {
    self->_prev = (Scalar**)malloc(child_size * sizeof(Scalar*));
    memcpy(self->_prev, child, child_size * sizeof(Scalar*));
  } else {
    self->_prev = NULL;
  }
  self->_backward = noop_backward;
  self->aux = 1;
  return self;
}

void add_backward(Scalar* self) {
  if (self->_prev_size == 2) {
    self->_prev[0]->grad += self->grad;
    self->_prev[1]->grad += self->grad;
  }
}

Scalar* add_val(Scalar* a, Scalar* b) {
  Scalar **child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  Scalar* out = initialize_scalars(a->data + b->data, child, 2);
  out->_backward = add_backward;
  return out;
}

void mul_backward(Scalar* self) {
  if (self->_prev_size == 2) {
    self->_prev[0]->grad += self->grad * self->_prev[1]->data;
    self->_prev[1]->grad += self->grad * self->_prev[0]->data;
  }
}

Scalar* mul_val(Scalar* a, Scalar* b) {
  Scalar **child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  Scalar* out = initialize_scalars(a->data * b->data, child, 2);
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double base = self->_prev[0]->data;
    double exponent = self->aux;
    self->_prev[0]->grad += self->grad * exponent * pow(base, exponent - 1);
  }
}

Scalar* pow_val(Scalar* a, float exp) {
  Scalar **child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(pow(a->data, exp), child, 1);
  out->aux = exp;
  out->_backward = pow_backward;
  return out;
}

void relu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    self->_prev[0]->grad += self->grad * (self->data > 0);
  }
}

Scalar* relu(Scalar* a) {
  Scalar **child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(a->data > 0 ? a->data : 0, child, 1);
  out->_backward = relu_backward;
  return out;
}

void tanh_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double tanh_data = self->data;
    self->_prev[0]->grad += self->grad * (1 - tanh_data * tanh_data);
  }
}

Scalar* tan_h(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(tanh(a->data), child, 1);
  out->_backward = tanh_backward;
  return out;
}

void sigmoid_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double sigmoid_data = self->data;
    self->_prev[0]->grad += self->grad * sigmoid_data * (1 - sigmoid_data);
  }
}

Scalar* sigmoid(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(1 / (1 + exp(-a->data)), child, 1);
  out->_backward = sigmoid_backward;
  return out;
}

void gelu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double x = self->_prev[0]->data;
    double tanh_out = tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)));
    double gelu_grad = 0.5 * (1 + tanh_out + x * (1 - pow(tanh_out, 2)) * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * pow(x, 2)));
    self->_prev[0]->grad += self->grad * gelu_grad;
  }
}

Scalar* gelu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  double gelu_data = 0.5 * a->data * (1 + tanh(sqrt(2 / M_PI) * (a->data + 0.044715 * pow(a->data, 3))));
  Scalar* out = initialize_scalars(gelu_data, child, 1);
  out->_backward = gelu_backward;
  return out;
}

void silu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double x = self->_prev[0]->data;
    double sigmoid_x = 1 / (1 + exp(-x));
    double silu_grad = sigmoid_x * (1 + x * (1 - sigmoid_x));
    self->_prev[0]->grad += self->grad * silu_grad;
  }
}

Scalar* silu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  double silu_data = a->data / (1 + exp(-a->data));
  Scalar* out = initialize_scalars(silu_data, child, 1);
  out->_backward = silu_backward;
  return out;
}

void swiglu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    double x = self->_prev[0]->data;
    double sigmoid_x = 1 / (1 + exp(-x));
    double tanh_out = tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)));
    double gelu_grad = 0.5 * (1 + tanh_out + x * (1 - pow(tanh_out, 2)) * sqrt(2 / M_PI) * (1 + 3 * 0.044715 * pow(x, 2)));
    double swiglu_grad = sigmoid_x * (gelu_grad + x * (1 - sigmoid_x) * gelu_grad);
    self->_prev[0]->grad += self->grad * swiglu_grad;
  }
}

Scalar* swiglu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  double swiglu_data = a->data * (1 / (1 + exp(-a->data))) * 0.5 * (1 + tanh(sqrt(2 / M_PI) * (a->data + 0.044715 * pow(a->data, 3))));
  Scalar* out = initialize_scalars(swiglu_data, child, 1);
  out->_backward = swiglu_backward;
  return out;
}

Scalar* negate(Scalar* a) {
  return mul_val(a, initialize_scalars(-1, NULL, 0));
}

Scalar* sub_val(Scalar* a, Scalar* b) {
  return add_val(a, negate(b));
}

Scalar* div_val(Scalar* a, Scalar* b) {
  return mul_val(a, pow_val(b, -1));
}

void print(Scalar* a) {
  std::cout<< "Scalar[data=(" << a->data << "), grad=(" << a->grad << ")]" << std::endl;
}

void dynamic_array_init(DynamicArray* array) {
  array->data = (Scalar**)malloc(10 * sizeof(Scalar*));
  array->size = 0;
  array->capacity = 10;
}

void dynamic_array_append(DynamicArray* array, Scalar* self) {
  if (array->size >= array->capacity) {
    array->capacity *= 2;
    array->data = (Scalar**)realloc(array->data, array->capacity * sizeof(Scalar*));
  }
  array->data[array->size++] = self;
}

void dynamic_array_free(DynamicArray* array) {
  free(array->data);
}

int dynamic_array_contains(DynamicArray* array, Scalar* self) {
  for (size_t i = 0; i < array->size; ++i) {
    if (array->data[i] == self) {
      return 1;
    }
  }
  return 0;
}

void build_topo(Scalar* self, DynamicArray* topo, DynamicArray* visited) {
  if (!dynamic_array_contains(visited, self)) {
    dynamic_array_append(visited, self);
    for (int i = 0; i < self->_prev_size; i++) {
      build_topo(self->_prev[i], topo, visited);
    }
    dynamic_array_append(topo, self);
  }
}

void cleanup(Scalar* v) {
  if (v->_prev != NULL) {
    free(v->_prev);
  }
  free(v);
}

void backward(Scalar* self) {
  self->grad = 1.0;

  DynamicArray visited;
  dynamic_array_init(&visited);
  DynamicArray topo;
  dynamic_array_init(&topo);

  build_topo(self, &topo, &visited);
  dynamic_array_free(&visited);

  for (int i = topo.size - 1; i >= 0; --i) {
    if (topo.data[i]->_backward != NULL) {
      topo.data[i]->_backward(topo.data[i]);
    }
  }

  dynamic_array_free(&topo);
}

double get_scalar_data(Scalar* v) {
  return v->data;
}

double get_scalar_grad(Scalar* v) {
  return v->grad;
}