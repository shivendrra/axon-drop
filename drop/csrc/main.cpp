#include "scalar.h"
#include <iostream>

int main() {
Scalar* x1 = initialize_scalars(2.0, DType::FLOAT32, nullptr, 0);
  Scalar* x2 = initialize_scalars(3.0, DType::FLOAT32, nullptr, 0);
  Scalar* x3 = initialize_scalars(5.0, DType::FLOAT32, nullptr, 0);
  Scalar* x4 = initialize_scalars(10.0, DType::FLOAT32, nullptr, 0);
  Scalar* x5 = initialize_scalars(1.0, DType::FLOAT32, nullptr, 0);
  Scalar* x6 = initialize_scalars(4.0, DType::FLOAT32, nullptr, 0);
  Scalar* x7 = initialize_scalars(-2.0, DType::FLOAT32, nullptr, 0);

  Scalar* a1 = add_val(x1, x2);            // a1 = x1 + x2
  Scalar* a2 = sub_val(x3, x4);            // a2 = x3 - x4
  Scalar* a3 = mul_val(a1, a2);            // a3 = a1 * a2
  Scalar* a4 = div_val(a3, x5);            // a4 = a3 / x5
  Scalar* a5 = pow_val(a4, 2.0);           // a5 = a4^2
  
  Scalar* a6 = sigmoid(a5);                // a6 = Sigmoid(a5)
  Scalar* a7 = tan_h(x7);                  // a7 = Tanh(x7)
  Scalar* a8 = relu(a3);                   // a8 = ReLU(a3)
  Scalar* a9 = silu(x6);                   // a9 = SiLU(x6)
  Scalar* a10 = gelu(x7);                  // a10 = GELU(x7)
  Scalar* a11 = swiglu(x2);                // a11 = SwiGLU(x2)

  Scalar* a12 = add_val(a6, a7);           // a12 = a6 + a7
  Scalar* a13 = add_val(a12, a8);          // a13 = a12 + a8
  Scalar* a14 = mul_val(a13, a9);          // a14 = a13 * a9
  Scalar* y = add_val(a14, a11);           // y = a14 + a11

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

  cleanup(x1);
  cleanup(x2);
  cleanup(x3);
  cleanup(x4);
  cleanup(x5);
  cleanup(x6);
  cleanup(x7);
  cleanup(a1);
  cleanup(a2);
  cleanup(a3);
  cleanup(a4);
  cleanup(a5);
  cleanup(a6);
  cleanup(a7);
  cleanup(a8);
  cleanup(a9);
  cleanup(a10);
  cleanup(a11);
  cleanup(a12);
  cleanup(a13);
  cleanup(a14);
  cleanup(y);

  return 0;
}