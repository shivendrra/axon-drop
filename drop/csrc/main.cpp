#include "value.h"
#include <iostream>

int main() {
  double data = 5.0;
  Value* val = create_value(&data);
  std::cout << "Value data: " << *(val->data) << std::endl;
  std::cout << "Value grad: " << *(val->grad) << std::endl;
  std::cout << "tanh: " << tan_h(&val) << std::endl;
  free(val->data);
  free(val->grad);
  free(val->exp);
  free(val->_prev_size);
  free(val);
  return 0;
}