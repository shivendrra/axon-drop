#ifndef __SCALAR_CUDA_KERNEL_H__
#define __SCALAR_CUDA_KERNEL_H__

__host__ void cpu_to_cuda(Scalar* a, int device_id);
__host__ void cuda_to_cpu(Scalar* a);
__host__ void free_cuda(float* data);

__global__ void add_scalar_cuda_kernel(float* a, float* b, float* out);
__host__ void add_scalar_cuda(Scalar* a, Scalar* b, float* out);
__global__ void sub_scalar_cuda_kernel(float* a, float* b, float* out);
__host__ void sub_scalar_cuda(Scalar* a, Scalar* b, float* out);
__global__ void mul_scalar_cuda_kernel(float* a, float* b, float* out);
__host__ void mul_scalar_cuda(Scalar* a, Scalar* b, float* out);
__global__ void div_scalar_cuda_kernel(float* a, float* b, float* out);
__host__ void div_scalar_cuda(Scalar* a, Scalar* b, float* out);
__global__ void pow_scalar_cuda_kernel(float* a, float exp, float* out);
__host__ void pow_scalar_cuda(Scalar* a, float exp, float* out);
__global__ void log_scalar_cuda_kernel(float* a, float* out);
__host__ void log_scalar_cuda(Scalar* a, float* out);
__global__ void sigmoid_scalar_cuda_kernel(float* a, float* out);
__host__ void sigmoid_scalar_cuda(Scalar* a, float* out);
__global__ void tanh_scalar_cuda_kernel(float* a, float* out);
__host__ void tanh_scalar_cuda(Scalar* a, float* out);
__global__ void sin_scalar_cuda_kernel(float* a, float* out);
__host__ void sin_scalar_cuda(Scalar* a, float* out);
__global__ void cos_scalar_cuda_kernel(float* a, float* out);
__host__ void cos_scalar_cuda(Scalar* a, float* out);
__global__ void relu_scalar_cuda_kernel(float* a, float* out);
__host__ void relu_scalar_cuda(Scalar* a, float* out);
__global__ void swiglu_scalar_cuda_kernel(float* a, float* out);
__host__ void swiglu_scalar_cuda(Scalar* a, float* out);
__global__ void equal_scalar_cuda_kernel(float* a, float* b, float* out);
__host__ void equal_scalar_cuda(Scalar* a, Scalar* b, float* out);

#endif