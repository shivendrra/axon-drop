#include "value.h"
#include "dtype.h"
#include <iostream>

int main() {
  double data1 = 2.0;
  double data2 = 3.0;
  double exp = 2.0;

  Value* val1 = initialize_value(&data1, DType::FLOAT32);
  Value* val2 = initialize_value(&data2, DType::FLOAT32);

  std::cout << "Initial values:" << std::endl;
  std::cout << "val1 data: " << get_data_as_double(val1) << ", grad: " << get_grad_as_double(val1) << std::endl;
  std::cout << "val2 data: " << get_data_as_double(val2) << ", grad: " << get_grad_as_double(val2) << std::endl;

  Value* add_result = add_val(val1, val2);
  std::cout << "Addition result: " << get_data_as_double(add_result) << std::endl;

  Value* mul_result = mul_val(val1, val2);
  std::cout << "Multiplication result: " << get_data_as_double(mul_result) << std::endl;

  Value* pow_result = pow_val(val1, &exp);
  std::cout << "Power result: " << get_data_as_double(pow_result) << std::endl;

  Value* relu_result = relu(val1);
  std::cout << "ReLU result: " << get_data_as_double(relu_result) << std::endl;

  backward(add_result);
  backward(mul_result);
  backward(pow_result);
  backward(relu_result);

  std::cout << "Gradients after backward pass:" << std::endl;
  std::cout << "val1 grad: " << get_grad_as_double(val1) << std::endl;
  std::cout << "val2 grad: " << get_grad_as_double(val2) << std::endl;

  free(val1->data);
  free(val1->grad);
  free(val1->exp);
  free(val1->_prev);
  free(val1->_prev_size);
  free(val1);

  free(val2->data);
  free(val2->grad);
  free(val2->exp);
  free(val2->_prev);
  free(val2->_prev_size);
  free(val2);

  free(add_result->data);
  free(add_result->grad);
  free(add_result->exp);
  free(add_result->_prev);
  free(add_result->_prev_size);
  free(add_result);

  free(mul_result->data);
  free(mul_result->grad);
  free(mul_result->exp);
  free(mul_result->_prev);
  free(mul_result->_prev_size);
  free(mul_result);

  free(pow_result->data);
  free(pow_result->grad);
  free(pow_result->exp);
  free(pow_result->_prev);
  free(pow_result->_prev_size);
  free(pow_result);

  free(relu_result->data);
  free(relu_result->grad);
  free(relu_result->exp);
  free(relu_result->_prev);
  free(relu_result->_prev_size);
  free(relu_result);

  return 0;
}