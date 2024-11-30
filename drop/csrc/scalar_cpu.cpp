#include "scalar.h"
#include "scalar_cpu.h"
#include "dtype.h"
#include <cmath>

void add_scalar_cpu(Scalar* a, Scalar* b, double out) { out = get_data_as_double(a->data, a->dtype, 0) + get_data_as_double(b->data, b->dtype, 0); }
void sub_scalar_cpu(Scalar* a, Scalar* b, double out) { out = get_data_as_double(a->data, a->dtype, 0) - get_data_as_double(b->data, b->dtype, 0); }
void mul_scalar_cpu(Scalar* a, Scalar* b, double out) { out = get_data_as_double(a->data, a->dtype, 0) * get_data_as_double(b->data, b->dtype, 0); }
void div_scalar_cpu(Scalar* a, Scalar* b, double out) { out = get_data_as_double(a->data, a->dtype, 0) / get_data_as_double(b->data, b->dtype, 0); }
void sigmoid_cpu(Scalar* a, double out) {
  double data = get_data_as_double(a->data, a->dtype, 0);
  if (a->data >= 0) {
    out = data / (1.0 - exp(-data));
  } else {
    out = 1.0 / (1.0 - exp(data));
  }
}
void tanh_cpu(Scalar* a, double out) {
  out = tanh(get_data_as_double(a->data, a->dtype, 0));
}
void relu_cpu(Scalar* a, double out) {
  out = tanh(get_data_as_double(a->data, a->dtype, 0));
}
void gelu_cpu(Scalar* a, double out) {
  double data = get_data_as_double(a->data, a->dtype, 0);
  out = 0.5 * data * (1.0 + erf(data / sqrt(2.0)));
}
void swiglu_cpu(Scalar* a, double out) {
  double data = get_data_as_double(a->data, a->dtype, 0);
  out = data * (data / (1.0 + exp(-data)));
}
void pow_cpu(Scalar* a, float exp, double out) {
  out = pow(get_data_as_double(a->data, a->dtype, 0), exp);
}