#ifndef SCALAR_CPU_H
#define SCALAR_CPU_H

#include "dtype.h"
#include "scalar.h"

void add_scalar_cpu(Scalar* a, Scalar* b, double out);
void sub_scalar_cpu(Scalar* a, Scalar* b, double out);
void mul_scalar_cpu(Scalar* a, Scalar* b, double out);
void div_scalar_cpu(Scalar* a, Scalar* b, double out);
void sigmoid_cpu(Scalar* a, double out);
void tanh_cpu(Scalar* a, double out);
void gelu_cpu(Scalar* a, double out);
void relu_cpu(Scalar* a, double out);

#endif