/*
  - scalar.cpp
  maininterface file for Scalar class
  - contains the initiliaztion, device switching logics for Scalar, switch-cases for device change,
  dtype & result changes.
  - also contains the main underlying autograd logic that works on scalar level, so we don't
  have to write `_backward()` function for each of Tensor's ops to compute gradient
*/

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include "scalar.h"
#include "dtype.h"

void noop_backward(Scalar *self) {}

Scalar* initialize_scalars(float data_value, DType dtype, Scalar** child, int child_size) {
  Scalar* self = (Scalar*)malloc(sizeof(Scalar));
  self->dtype = dtype;
  self->data = initialize_data(data_value, dtype);
  self->grad = initialize_data(0.0, dtype);

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
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype));
    set_data_from_float(self->_prev[1]->grad, self->_prev[1]->dtype, get_data_as_float(self->_prev[1]->grad, self->_prev[1]->dtype) + get_data_as_float(self->grad, self->dtype));
  }
}

Scalar* add_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  float result = get_data_as_float(a->data, a->dtype) + get_data_as_float(b->data, b->dtype);
  Scalar* out = initialize_scalars(result, a->dtype, child, 2);
  out->_backward = add_backward;
  return out;
}

void mul_backward(Scalar* self) {
  if (self->_prev_size == 2) {
    float a = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float b = get_data_as_float(self->_prev[1]->data, self->_prev[1]->dtype);
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * b);
    set_data_from_float(self->_prev[1]->grad, self->_prev[1]->dtype, get_data_as_float(self->_prev[1]->grad, self->_prev[1]->dtype) + get_data_as_float(self->grad, self->dtype) * a);
  }
}

Scalar* mul_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  float result = get_data_as_float(a->data, a->dtype) * get_data_as_float(b->data, b->dtype);
  Scalar* out = initialize_scalars(result, a->dtype, child, 2);
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float base = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float exponent = self->aux;
    float grad = exponent * pow(base, exponent - 1);
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* pow_val(Scalar* a, float exp) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = pow(get_data_as_float(a->data, a->dtype), exp);
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->aux = exp;
  out->_backward = pow_backward;
  return out;
}

void log_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float a = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float grad = 1.0 / a;
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* log_val(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = log(get_data_as_float(a->data, a->dtype));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = log_backward;
  return out;
}

void relu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float grad = (get_data_as_float(self->data, self->dtype) > 0) ? get_data_as_float(self->grad, self->dtype) : 0.0;
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + grad);
  }
}

Scalar* relu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = fmax((float)0.0, get_data_as_float(a->data, a->dtype));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = relu_backward;
  return out;
}

void tanh_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float tanh_data = get_data_as_float(self->data, self->dtype);
    float grad = 1.0 - tanh_data * tanh_data;
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* tan_h(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = tanh(get_data_as_float(a->data, a->dtype));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = tanh_backward;
  return out;
}

void cos_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float grad = -sinf(get_data_as_float(self->data, self->dtype));
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* cos_val(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = cosf(get_data_as_float(a->data, a->dtype));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = tanh_backward;
  return out;
}

void sin_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float grad = cosf(get_data_as_float(self->data, self->dtype));
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* sin_val(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = sinf(get_data_as_float(a->data, a->dtype));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = tanh_backward;
  return out;
}

void sigmoid_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float sigmoid_data = get_data_as_float(self->data, self->dtype);
    float grad = sigmoid_data * (1.0 - sigmoid_data);
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* sigmoid(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = 1.0 / (1.0 + exp(-get_data_as_float(a->data, a->dtype)));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = sigmoid_backward;
  return out;
}

void silu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float x = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float sigmoid_x = 1.0 / (1.0 + exp(-x));
    float grad = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x));
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* silu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float x = get_data_as_float(a->data, a->dtype);
  float result = x / (1.0 + exp(-x));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = silu_backward;
  return out;
}

void gelu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float x = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float grad = 0.5 * (1.0 + erf(x / sqrt(2.0))) + (x * exp(-x * x / 2.0) / sqrt(2.0 * PI_VAL));
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* gelu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float x = get_data_as_float(a->data, a->dtype);
  float result = 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = gelu_backward;
  return out;
}

void swiglu_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    float x = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float grad = x / (1.0 + exp(-x));
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) * grad);
  }
}

Scalar* swiglu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float x = get_data_as_float(a->data, a->dtype);
  float result = x * (x / (1.0 + exp(-x)));
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = swiglu_backward;
  return out;
}

void negate_backward(Scalar* self) {
  if (self->_prev_size == 1) {
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) - get_data_as_float(self->grad, self->dtype));
  }
}

Scalar* negate(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  float result = -get_data_as_float(a->data, a->dtype);
  Scalar* out = initialize_scalars(result, a->dtype, child, 1);
  out->_backward = negate_backward;
  return out;
}

Scalar* sub_val(Scalar* a, Scalar* b) {
  return add_val(a, negate(b));
}

void div_backward(Scalar* self) {
  if (self->_prev_size == 2) {
    float a = get_data_as_float(self->_prev[0]->data, self->_prev[0]->dtype);
    float b = get_data_as_float(self->_prev[1]->data, self->_prev[1]->dtype);
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype) + get_data_as_float(self->grad, self->dtype) / b);
    set_data_from_float(self->_prev[1]->grad, self->_prev[1]->dtype, get_data_as_float(self->_prev[1]->grad, self->_prev[1]->dtype) - get_data_as_float(self->grad, self->dtype) * a / (b * b));
  }
}

Scalar* div_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  float result = get_data_as_float(a->data, a->dtype) / get_data_as_float(b->data, b->dtype);
  Scalar* out = initialize_scalars(result, a->dtype, child, 2);
  out->_backward = div_backward;
  return out;
}

void equal_backward(Scalar* self) {
  if (self->_prev_size == 2) {
    set_data_from_float(self->_prev[0]->grad, self->_prev[0]->dtype, get_data_as_float(self->_prev[0]->grad, self->_prev[0]->dtype));
    set_data_from_float(self->_prev[1]->grad, self->_prev[1]->dtype, get_data_as_float(self->_prev[1]->grad, self->_prev[1]->dtype));
  }
}

Scalar* equal_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  float result = (get_data_as_float(a->data, a->dtype) == get_data_as_float(b->data, b->dtype)) ? 1.0f : 0.0f;
  Scalar* out = initialize_scalars(result, a->dtype, child, 2);
  out->_backward = mul_backward;
  return out;
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

void backward(Scalar* self) {
  set_data_from_float(self->grad, self->dtype, 1.0);

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

void print(Scalar* v) {
  printf("Value: %f, Grad: %f\n, dtype=drop.%s\n", get_data_as_float(v->data, v->dtype), get_data_as_float(v->data, v->dtype), dtype_to_string(v->dtype));
}

void cleanup(Scalar* v) {
  if (v->_prev != NULL) {
    free(v->_prev);
  }
  free(v);
}

float get_scalar_data(Scalar* v) {
  return get_data_as_float(v->data, v->dtype);
}

float get_scalar_grad(Scalar* v) {
  return get_data_as_float(v->grad, v->dtype);
}

void set_scalar_data(Scalar* v, float value) {
  v->data = initialize_data(value, v->dtype);
}

void set_scalar_grad(Scalar* v, float value) {
  v->grad = initialize_data(value, v->dtype);
}