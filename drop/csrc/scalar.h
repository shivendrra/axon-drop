/* 
  - scalar.h header file for scalar.cpp & Scalar
  - scalar value autograd performing ops over a single value at a time
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libscalar.so scalar.cpp dtype.cpp
    -- '.dll': g++ -shared -o libscalar.dll scalar.cpp dtype.cpp
*/

#ifndef __SCALAR_H__
#define __SCALAR_H__

#include <stdlib.h>
#include "dtype.h"

#define PI_VAL 3.14159f  // defining pi using preprocessor directive

typedef struct Scalar {
  void* data;                           // single value for operation
  void* grad;                           // grad related to the data
  DType dtype;                          // dtype identifier from dtype.cpp
  struct Scalar** _prev;                // struct to hold previous/child Scalars
  int _prev_size;                       // sets the size of prev struct
  void (*_backward)(struct Scalar*);    // backward function for autograd
  float aux;                            // auxillary value, sometimes needed
} Scalar;

// dynamic array struct to manage dynamic scalar storage for autograd
typedef struct DynamicArray {
  Scalar** data;
  size_t size;
  size_t capacity;
} DynamicArray;

extern "C" {
  Scalar* initialize_scalars(float data, DType dtype, Scalar** child, int child_size);
  void noop_backward(Scalar* v);

  Scalar* add_val(Scalar* a, Scalar* b);
  void add_backward(Scalar* v);
  Scalar* mul_val(Scalar* a, Scalar* b);
  void mul_backward(Scalar* v);
  Scalar* pow_val(Scalar* a, float exp);
  void pow_backward(Scalar* v);
  Scalar* log_val(Scalar* a);
  void log_backward(Scalar* v);

  Scalar* negate(Scalar* a);
  Scalar* sub_val(Scalar* a, Scalar* b);
  Scalar* div_val(Scalar* a, Scalar* b);
  Scalar* equal_val(Scalar* a, Scalar* b);

  Scalar* relu(Scalar* a);
  void relu_backward(Scalar* v);
  Scalar* sigmoid(Scalar* a);
  void sigmoid_backward(Scalar* v);
  Scalar* tan_h(Scalar* a);
  void tanh_backward(Scalar* v);
  Scalar* sin_val(Scalar* a);
  void sin_backward(Scalar* v);
  Scalar* cos_val(Scalar* a);
  void cos_backward(Scalar* v);
  Scalar* silu(Scalar* a);
  void silu_backward(Scalar* v);
  Scalar* gelu(Scalar* a);
  void gelu_backward(Scalar* v);
  Scalar* swiglu(Scalar* a);
  void swiglu_backward(Scalar* v);

  void build_topo(Scalar* v, DynamicArray* topo, DynamicArray* visited);
  void backward(Scalar* v);
  void print(Scalar* v);
  void cleanup(Scalar* v);
  float get_scalar_data(Scalar* v);
  float get_scalar_grad(Scalar* v);
  void set_scalar_data(Scalar* v, float value);
  void set_scalar_grad(Scalar* v, float value);
}

#endif