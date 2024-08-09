#include "scalar.h"
#include <iostream>

int main() {
  Scalar* x1 = initialize_scalars(2, NULL, 0);
  Scalar* x2 = initialize_scalars(3, NULL, 0);
  Scalar* x3 = initialize_scalars(5, NULL, 0);
  Scalar* x4 = initialize_scalars(10, NULL, 0);
  Scalar* x5 = initialize_scalars(1, NULL, 0);
  Scalar* x6 = initialize_scalars(4, NULL, 0);
  Scalar* x7 = initialize_scalars(-2, NULL, 0);

  Scalar* a1 = add_val(x1, x2);            // a1 = x1 + x2
  Scalar* a2 = sub_val(x3, x4);            // a2 = x3 - x4
  Scalar* a3 = mul_val(a1, a2);            // a3 = a1 * a2
  Scalar* a4 = pow_val(a3, 2);             // a4 = a3^2
  Scalar* a5 = mul_val(x5, x6);            // a5 = x5 * x6
  Scalar* a6 = sigmoid(a5);                // a6 = Sigmoid(a5)
  Scalar* a7 = tan_h(x7);                  // a7 = Tanh(x7)
  Scalar* a8 = add_val(a4, a6);            // a8 = a4 + a6
  Scalar* a9 = add_val(a8, a7);            // a9 = a8 + a7
  Scalar* y = relu(a9);                    // y = ReLU(a9)

  std::cout << "Before backward pass:" << std::endl;
  print(x1);
  print(x2);
  print(x3);
  print(x4);
  print(x5);
  print(x6);
  print(x7);
  print(y);

  backward(y);

  std::cout << "After backward pass:" << std::endl;
  print(x1);
  print(x2);
  print(x3);
  print(x4);
  print(x5);
  print(x6);
  print(x7);
  print(y);

  return 0;
}