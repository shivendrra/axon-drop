/*
  cuda.h
  - header file for cuda.cu, all the gpu kernels
  - conatains all the cuda related functions, from changing of the device to computing acivations
  - to be included with ``tensor.cpp`` & ``dtype.cpp`` when compiled
*/

#ifndef __CUDA_KERNEL_H__
#define __CUDA_KERNEL_H__

__host__ void cpu_to_cuda(Tensor* tensor, int device_id);
__host__ void cuda_to_cpu(Tensor* tensor);
__host__ void free_cuda(Scalar* data);

__global__ void add_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int size);
__host__ void add_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void sub_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int size);
__host__ void sub_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void mul_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int size);
__host__ void mul_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void div_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int size);
__host__ void div_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void scalar_div_tensor_cuda_kernel(float scalar, Scalar* a, Scalar* out, int size);
__host__ void scalar_div_tensor_cuda(float scalar, Tensor* a, Tensor* out);
__global__ void tensor_div_scalar_cuda_kernel(foat* a, float scalar, Scalar* out, int size);
__host__ void tensor_div_scalar_cuda(Tensor* a, float scalar, Tensor* out);
__global__ void scalar_mul_tensor_cuda_kernel(float scalar, Scalar* a, Scalar* out, int size);
__host__ void scalar_mul_tensor_cuda(float scalar, Tensor* a, Tensor* out);

__global__ void add_broadcasted_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
__host__ void add_broadcasted_tensor_cuda(Tensor* a, Tensor* b, Scalar* out, int* broadcasted_shape, int broadcasted_size);
__global__ void sub_broadcasted_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
__host__ void sub_broadcasted_tensor_cuda(Tensor* a, Tensor* b, Scalar* out, int* broadcasted_shape, int broadcasted_size);
__global__ void mul_broadcasted_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
__host__ void mul_broadcasted_tensor_cuda(Tensor* a, Tensor* b, Scalar* out, int* broadcasted_shape, int broadcasted_size);
__global__ void div_broadcasted_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
__host__ void div_broadcasted_tensor_cuda(Tensor* a, Tensor* b, Scalar* out, int* broadcasted_shape, int broadcasted_size);

__global__ void tensor_pow_scalar_cuda_kernel(Scalar* a, float exponent, Scalar* out, int size);
__host__ void tensor_pow_scalar_cuda(Tensor* a, float exponent Tensor* out);
__global__ void scalar_pow_tensor_cuda_kernel(float base, Scalar* data, Scalar* out, int size);
__host__ void scalar_pow_tensor_cuda(float base, Tensor* data, Tensor* out);
__global__ void log_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void log_tensor_cuda(Tensor* a, Tensor* out);
__global__ void sigmoid_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void sigmoid_tensor_cuda(Tensor* a, Tensor* out);
__global__ void sin_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void sin_tensor_cuda(Tensor* a, Tensor* out);
__global__ void cos_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void cos_tensor_cuda(Tensor* a, Tensor* out);
__global__ void tanh_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void tanh_tensor_cuda(Tensor* a, Tensor* out);
__global__ void relu_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void relu_tensor_cuda(Tensor* a, Tensor* out);

__global__ void sum_tensor_cuda_kernel(Scalar* a, Tensor* out);
__global__ void sum_tensor_cuda_kernel_axis(Scalar* a, Scalar* out, int* strides, int* shape, int axis, int ndim, int axis_strides, int size, int res_size);
__host__ void sum_tensor_cuda(Scalar* a, Scalar* out, int axis);
__global__ void max_tensor_cuda_kernel(Scalar* a, Tensor* out);
__global__ void max_tensor_cuda_kernel_axis(Scalar* a, Scalar* out, int* strides, int* shape, int axis, int ndim, int axis_strides, int size, int res_size);
__host__ void max_tensor_cuda(Scalar* a, Scalar* out, int axis);
__global__ void min_tensor_cuda_kernel(Scalar* a, Tensor* out);
__global__ void min_tensor_cuda_kernel_axis(Scalar* a, Scalar* out, int* strides, int* shape, int axis, int ndim, int axis_strides, int size, int res_size);
__host__ void min_tensor_cuda(Scalar* a, Scalar* out, int axis);

__global__ void equal_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int size);
__host__ void equal_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void zeros_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void zeros_tensor_cuda(Tensor* a, Tensor* out);
__global__ void ones_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void ones_tensor_cuda(Tensor* a, Tensor* out);
__global__ void assign_tensor_cuda_kernel(Scalar* a, Scalar* out, int size);
__host__ void assign_tensor_cuda(Tensor* a, Tensor* out);
__global__ void equal_broadcasted_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int* broadcasted_shape, int* strides1, int* strides2, int max_ndim, int size);
__host__ void equal_broadcasted_tensor_cuda(Tensor* a, Tensor* b, Scalar* out, int* broadcasted_shape, int broadcasted_size);
__global__ void make_contiguous_tensor_cuda_kernel(Scalar* a, Scalar* out, int ndim, int size, int* strides, int* new_strides);
__host__ void make_contiguous_tensor_cuda(Tensor* a, Scalar* out, int* new_strides);

__global__ void transpose_1d_tensor_cuda_kernel(Scalar* a, Scalar* out, int rows, int cols);
__host__ void tranpsose_1d_tensor_cuda(Tensor* a, Tensor* out);
__global__ void transpose_2d_tensor_cuda_kernel(Scalar* a, Scalar* out, int rows, int cols);
__host__ void tranpsose_2d_tensor_cuda(Tensor* a, Tensor* out);
__global__ void transpose_3d_tensor_cuda_kernel(Scalar* a, Scalar* out, int batch, int rows, int cols);
__host__ void tranpsose_3d_tensor_cuda(Tensor* a, Tensor* out);

__global__ void matmul_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int rows1, int cols1, int cols2);
__host__ void matmul_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void batched_matmul_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int batch_size, int rows1, int cols1, int cols2);
__host__ void batched_matmul_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);
__global__ void broadcasted_batched_matmul_tensor_cuda_kernel(Scalar* a, Scalar* b, Scalar* out, int batch_size, int rows1, int cols1, int cols2);
__host__ void broadcasted_batched_matmul_tensor_cuda(Tensor* a, Tensor* b, Tensor* out);

#endif