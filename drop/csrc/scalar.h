#ifndef SCALAR_H
#define SCALAR_H

typedef struct Scalar {
  double data;
  double grad;
  struct Scalar** _prev;
  void (*_backward)(struct Scalar*);
} Scalar;

#endif