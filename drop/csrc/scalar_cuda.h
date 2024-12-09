#ifndef __SCALAR_CUDA_KERNEL_H__
#define __SCALAR_CUDA_KERNEL_H__

__host__ voicd cpu_to_cuda(Scalar* a, int device_id);
__host__ void cuda_to_cpu(Scalar* a);
__host__ void free_cuda(float* data);

#endif