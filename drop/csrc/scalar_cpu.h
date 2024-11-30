/*
 - scalar_cpu.h header for scalar's cpu version functions
 - imported in ``scalar.cpp``
*/

#ifndef SCALAR_CPU_H
#define SCALAR_CPU_H

#include "dtype.h"
#include "scalar.h"

void add_scalar_cpu(Scalar* a, Scalar* b, float out);
void sub_scalar_cpu(Scalar* a, Scalar* b, float out);
void mul_scalar_cpu(Scalar* a, Scalar* b, float out);
void div_scalar_cpu(Scalar* a, Scalar* b, float out);
void sigmoid_cpu(Scalar* a, float out);
void tanh_cpu(Scalar* a, float out);
void gelu_cpu(Scalar* a, float out);
void relu_cpu(Scalar* a, float out);
void swiglu_cpu(Scalar* a, float out);
void pow_cpu(Scalar* a, float exp, float out);

#endif