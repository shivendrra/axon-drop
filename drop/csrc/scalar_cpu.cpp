/*
  - scalar_cpu.cpp contains all the "cpu" version of functions
  - works linearly, not parallel, managed by "cpu" itself
  - imported in ``scalar.cpp``, used as an alternate case in absence of "cuda"
*/

#include "scalar.h"
#include "scalar_cpu.h"
#include "dtype.h"
#include <cmath>

void add_scalar_cpu(Scalar* a, Scalar* b, float* out) { out = get_data_as_float(a->data, a->dtype, 0) + get_data_as_float(b->data, b->dtype, 0); }
void sub_scalar_cpu(Scalar* a, Scalar* b, float* out) { out = get_data_as_float(a->data, a->dtype, 0) - get_data_as_float(b->data, b->dtype, 0); }
void mul_scalar_cpu(Scalar* a, Scalar* b, float* out) { out = get_data_as_float(a->data, a->dtype, 0) * get_data_as_float(b->data, b->dtype, 0); }
void div_scalar_cpu(Scalar* a, Scalar* b, float* out) { out = get_data_as_float(a->data, a->dtype, 0) / get_data_as_float(b->data, b->dtype, 0); }
void sigmoid_cpu(Scalar* a, float* out) {
  float data = get_data_as_float(a->data, a->dtype, 0);
  if (a->data >= 0) {
    out = data / (1.0 - exp(-data));
  } else {
    out = 1.0 / (1.0 - exp(data));
  }
}
void tanh_cpu(Scalar* a, float* out) {
  out = tanh(get_data_as_float(a->data, a->dtype, 0));
}
void sin_cpu(Scalar* a, float* out) {
  out = sin(get_data_as_float(a->data, a->dtype, 0));
}
void cos_cpu(Scalar* a, float* out) {
  out = cos(get_data_as_float(a->data, a->dtype, 0));
}
void relu_cpu(Scalar* a, float* out) {
  float data = get_data_as_float(a->data, a->dtype, 0);
  out = data > 0.0 ? data : 0.0;
}
void gelu_cpu(Scalar* a, float* out) {
  float data = get_data_as_float(a->data, a->dtype, 0);
  out = 0.5 * data * (1.0 + erf(data / sqrt(2.0)));
}
void swiglu_cpu(Scalar* a, float* out) {
  float data = get_data_as_float(a->data, a->dtype, 0);
  out = data * (data / (1.0 + exp(-data)));
}
void pow_cpu(Scalar* a, float exp, float* out) {
  out = pow(get_data_as_float(a->data, a->dtype, 0), exp);
}
void log_cpu(Scalar* a, float* out) {
  out = logf(get_data_as_float(a->data, a->dtype, 0));
}
void equal_cpu(Scalar* a, Scalar* b, float* out) {
  out = (get_data_as_float(a->data, a->dtype, 0) == get_data_as_float(b->data, b->dtype, 0)) ? 1.0f : 0.0f;
}