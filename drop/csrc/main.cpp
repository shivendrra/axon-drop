#include "tensor.h"
#include <iostream>

int main() {
  // Initialize tensor shapes and data
  int* shape1 = (int*)malloc(2 * sizeof(int));
  int* shape2 = (int*)malloc(2 * sizeof(int));
  shape1[0] = 2, shape1[1] = 2;
  shape2[0] = 2, shape2[1] = 2;

  float* data1 = (float*)malloc(4 * sizeof(float));
  float* data2 = (float*)malloc(4 * sizeof(float));
  data1[0] = 1.0; data1[1] = 2.0; data1[2] = 3.0; data1[3] = 4.0;
  data2[0] = 5.0; data2[1] = 6.0; data2[2] = 7.0; data2[3] = 8.0;

  // Create tensors
  DType dtype = DType::FLOAT32;
  Tensor* t1 = create_tensor(data1, shape1, 2, dtype);
  Tensor* t2 = create_tensor(data2, shape2, 2, dtype);
  printf("\nT1:");
  print_tensor(t1);
  printf("\nT2:");
  print_tensor(t2);

  // Perform operations
  Tensor* t3 = add_tensor(t1, t2);         // Element-wise addition
  Tensor* t4 = elemwise_mul_tensor(t1, t2); // Element-wise multiplication
  // Tensor* t5 = matmul_tensor(t1, t2);      // Matrix multiplication (2x2)
  // Tensor* t6 = transpose_tensor(t5);  // Transposing tensor
  Tensor* t7 = sum_tensor(t4, -1, false);

  // Print tensors
  printf("\nT3 (T1 + T2):");
  print_tensor(t3);
  printf("\nT4 (T1 * T2):");
  print_tensor(t4);
  // printf("\nT5 (T1 @ T2):");
  // print_tensor(t5);
  // printf("\nT6 (T5)^T:");
  // print_tensor(t6);
  printf("\nT7 sum(T6):");
  print_tensor(t7);

  // Perform autograd
  printf("\n Performing backward pass on T7: ");
  tensor_backward(t7);

  printf("\ngrads:\n");
  print_grads(t1);
  print_grads(t2);

  // Clean up memory
  delete_tensor(t1);
  delete_tensor(t2);
  delete_tensor(t3);
  delete_tensor(t4);
  // delete_tensor(t5);

  return 0;
}
