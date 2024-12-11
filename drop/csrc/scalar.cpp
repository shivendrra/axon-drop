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
#include "scalar_cpu.h"
#include "dtype.h"
#include "scalar_cuda.h"

void noop_backward(Scalar *self) {}

Scalar* initialize_scalars(float data_value, DType dtype, Scalar** child, int child_size, char* device) {
  Scalar* self = (Scalar*)malloc(sizeof(Scalar));
  if (!self) {
    fprintf(stderr, "Memory allocation for Scalar Failed!\n");
    exit(1);
  }

  self->dtype = dtype;
  self->data = initialize_data(data_value, dtype);
  self->grad = initialize_data(0.0, dtype);
  self->_prev_size = child_size;
  self->device = (char*)malloc(strlen(device) + 1);
  if (device != NULL) {
    strcpy(self->device, device);
  } else {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
  if (child_size > 0) {
    self->_prev = (Scalar**)malloc(child_size * sizeof(Scalar*));
    if (!self->_prev) {
      fprintf(stderr, "Failed to allocate memory for _prev!\n");
      exit(1);
    }
    memcpy(self->_prev, child, child_size * sizeof(Scalar*));
  } else {
    self->_prev = NULL;
  }
  self->_backward = noop_backward;
  self->aux = 1;
  return self;
}

void scalar_to_device(Scalar* a, char* device) {
  int device_id = 0;
  char *endptr, *device_type;
  long num = strtol(device, &endptr, 10);
  if (*endptr == '\0') {
    device_id = (int)num;
    device_type = new char[strlen("cuda") + 1];
    strcpy(device_type, "cuda");
  } else {
    device_type = new char[strlen("cpu") + 1];
    strcpy(device_type, "cpu");
  }

  if((strcmp(device_type, "cuda") == 0) && (strcmp(a->device, "cpu") == 0)) {
    cpu_to_cuda(a, device_id);
  } else if ((strcmp(device_type, "cpu") == 0) && (strcmp(a->device, "cuda") == 0)) {
    cuda_to_cpu(a);
  }
  free(device_type);
}

float get_scalar_data(Scalar* v) {
  float data = get_data_as_float(v->data, v->dtype, 0);
  return data;
}

float get_scalar_grad(Scalar* v) {
  float grad = get_data_as_float(v->grad, v->dtype, 0);
  return grad;
}

void set_scalar_data(Scalar* v, float value) {
  v->data = initialize_data(value, v->dtype);
}

void set_scalar_grad(Scalar* v, float value) {
  v->grad = initialize_data(value, v->dtype);
}

void cleanup(Scalar* v) {
  if (v->_prev != NULL) {
    free(v->_prev);
  }
  free(v);
}

void print(Scalar* v) {
  if (!v) {
    fprintf(stderr, "Scalar is null\n");
    exit(1);
  }
  std::cout << "Value: " << get_data_as_float(v->data, v->dtype, 0)
            << ", Grad: " << get_data_as_float(v->grad, v->dtype, 0) << std::endl;
}

Scalar* add_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Scalars must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float out = (float*)malloc(sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed for output scalar\n");
      exit(1);
    }
    add_scalar_cpu(a, b, out);
    return initialize_scalars(out, a->dtype, child, 2, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Scalar* mul_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Scalars must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float out = (float*)malloc(sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed for output scalar\n");
      exit(1);
    }
    mul_scalar_cpu(a, b, out);
    return initialize_scalars(out, a->dtype, child, 2, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Scalar* sub_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Scalars must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float out = (float*)malloc(sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed for output scalar\n");
      exit(1);
    }
    sub_scalar_cpu(a, b, out);
    return initialize_scalars(out, a->dtype, child, 2, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Scalar* div_val(Scalar* a, Scalar* b) {
  Scalar** child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Scalars must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float out = (float*)malloc(sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed for output scalar\n");
      exit(1);
    }
    div_scalar_cpu(a, b, out);
    return initialize_scalars(out, a->dtype, child, 2, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}