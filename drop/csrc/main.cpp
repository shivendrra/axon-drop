#include "tensor.h"
#include <iostream>

int main() {
  // Initialize tensor shapes and data
  int* shape1 = (int*)malloc(2 * sizeof(int));
  int* shape2 = (int*)malloc(2 * sizeof(int));
  shape1[0] = 2, shape1[1] = 4;
  shape2[0] = 2, shape2[1] = 4;

  float* data1 = (float*)malloc(8 * sizeof(float));
  float* data2 = (float*)malloc(8 * sizeof(float));
  data1[0] = 2.0; data1[1] = 4.0; data1[2] = 5.0; data1[3] = -4.0;
  data1[4] = -3.0; data1[5] = 0.0; data1[6] = 9.0; data1[7] = -1.0;
  data2[0] = 1.0; data2[1] = 0.0; data2[2] = -2.0; data2[3] = 0.0;
  data2[4] = -1.0; data2[5] = 10.0; data2[6] = -2.0; data2[7] = 4.0;

  // Create tensors
  DType dtype = DType::FLOAT32;
  Tensor* t1 = create_tensor(data1, shape1, 2, dtype);
  Tensor* t2 = create_tensor(data2, shape2, 2, dtype);
  printf("\nT1:");
  print_tensor(t1);
  printf("\nT2:");
  print_tensor(t2);

  Scalar* s1 = initialize_scalars(2.0, DType::FLOAT32, NULL, 0);

  // Perform operations
  Tensor* t3 = add_tensor(t1, t2);
  Tensor* t4 = tanh_tensor(t3);
  Tensor* t5 = silu_tensor(t4);
  Tensor* t6 = tensor_pow_scalar(t5, s1);
  Tensor* t7 = sigmoid_tensor(t6);
  Tensor* t8 = sum_tensor(t7, -1, false);

  // Perform autograd
  printf("\n Performing backward pass on T7: ");
  tensor_backward(t8);

  printf("\ngrads:\n");
  print_grads(t1);
  print_grads(t2);
  // Print tensors
  printf("\nT3 (T1 + T2):");
  print_grads(t3);
  printf("\nT4 (T1 * T2):");
  print_grads(t4);
  printf("\nT5 (T1 @ T2):");
  print_grads(t5);
  printf("\nT6 (T5)^T:");
  print_grads(t6);
  printf("\nT7 sum(T6):");
  print_grads(t7);
  printf("\nSum t7:");
  print_grads(t8);

  // Clean up memory
  delete_tensor(t1);
  delete_tensor(t2);
  delete_tensor(t3);
  delete_tensor(t4);
  delete_tensor(t5);
  delete_tensor(t6);
  delete_tensor(t7);
  delete_tensor(t8);

  return 0;
}
