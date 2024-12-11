#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "scalar.h"
#include "dtype.h"
#include "scalar_cuda.h"

__host__ void cpu_to_cuda(Scalar* a, int device_id) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (device_id >= deviceCount) {
    fprintf(stderr, "Could not send tensor to device %d, only %d devices available\n", device_id, deviceCount);
    exit(1);
  }
  cudaSetDevice(device_id);
  float* d_tmp;
  cudaMalloc(&d_tmp, sizeof(float));
  cudaMemcpy(d_tmp, a->data, sizeof(float), cudaMemcpyHostToDevice);
  a->data = d_tmp;
  a->device = (char*)malloc(strlen("cuda") + 1);
  strcpy(a->device, "cuda");
}

__host__ void cuda_to_cpu(Scalar* a) {
  float* d_tmp = malloc(sizeof(float));
  cudaMemcpy(d_tmp, a->data, sizeof(float), cudaMemcpyHostToDevice);
  cudaFree(a->data);
  a->data = d_tmp;
  a->device = (char*)malloc(strlen("cpu") + 1);
  strcpy(a->device, "cpu");
}

__host__ void free_cuda(float* data) {
  cudaFree(data);
}

__global__ void add_scalar_cuda_kernel(float* a, float* b, float* out) {
  *out = *a + *b;
}

__global__ void sub_scalar_cuda_kernel(float* a, float* b, float* out) {
  *out = *a - *b;
}

__global__ void mul_scalar_cuda_kernel(float* a, float* b, float* out) {
  *out = (*a) * (*b);
}

__global__ void div_scalar_cuda_kernel(float* a, float* b, float* out) {
  *out = (*a) / (*b);
}

__host__ void add_scalar_cuda(Scalar* a, Scalar* b, float* out) {
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  float val_a = get_data_as_float(a->data, a->dtype, 0);
  float val_b = get_data_as_float(b->data, b->dtype, 0);
  
  cudaMemcpy(d_a, &val_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &val_b, sizeof(float), cudaMemcpyHostToDevice);

  add_scalar_cuda_kernel<<<1, 1>>>(d_a, d_b, d_out);
  
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

__host__ void sub_scalar_cuda(Scalar* a, Scalar* b, float* out) {
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  float val_a = get_data_as_float(a->data, a->dtype, 0);
  float val_b = get_data_as_float(b->data, b->dtype, 0);

  cudaMemcpy(d_a, &val_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &val_b, sizeof(float), cudaMemcpyHostToDevice);

  sub_scalar_cuda_kernel<<<1, 1>>>(d_a, d_b, d_out);
  
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

__host__ void mul_scalar_cuda(Scalar* a, Scalar* b, float* out) {
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  float val_a = get_data_as_float(a->data, a->dtype, 0);
  float val_b = get_data_as_float(b->data, b->dtype, 0);

  cudaMemcpy(d_a, &val_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &val_b, sizeof(float), cudaMemcpyHostToDevice);

  mul_scalar_cuda_kernel<<<1, 1>>>(d_a, d_b, d_out);
  
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

__host__ void div_scalar_cuda(Scalar* a, Scalar* b, float* out) {
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  float val_a = get_data_as_float(a->data, a->dtype, 0);
  float val_b = get_data_as_float(b->data, b->dtype, 0);

  cudaMemcpy(d_a, &val_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &val_b, sizeof(float), cudaMemcpyHostToDevice);

  div_scalar_cuda_kernel<<<1, 1>>>(d_a, d_b, d_out);
  
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

__global__ void pow_scalar_cuda_kernel(float* a, float exp, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = powf(a[0], exp);
  }
}

__host__ void pow_scalar_cuda(Scalar* a, float exp, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  
  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  pow_scalar_cuda_kernel<<<1, 1>>>(d_a, exp, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void log_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = logf(a[0]);
  }
}

__host__ void log_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  log_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void sigmoid_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = 1.0f / (1.0f + expf(-a[0]));
  }
}

__host__ void sigmoid_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  sigmoid_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void tanh_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = tanhf(a[0]);
  }
}

__host__ void tanh_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  
  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  tanh_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void sin_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = sin(a[0]);
  }
}

__host__ void sin_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  
  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  sin_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void cos_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = cos(a[0]);
  }
}

__host__ void cos_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  
  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  cos_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void relu_scalar_cuda_kernel(float* a, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = (a[0] >= 0) ? a[0] : 0.0f;
  }
}

__host__ void relu_scalar_cuda(Scalar* a, float* out) {
  float *d_a, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  
  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  relu_scalar_cuda_kernel<<<1, 1>>>(d_a, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_out);
}

__global__ void equal_scalar_cuda_kernel(float* a, float* b, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) {
    out[0] = (a[0] == b[0]) ? 1.0f : 0.0f;
  }
}

__host__ void equal_scalar_cuda(Scalar* a, Scalar* b, float* out) {
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_out, sizeof(float));

  cudaMemcpy(d_a, a->data, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b->data, sizeof(float), cudaMemcpyHostToDevice);
  equal_scalar_cuda_kernel<<<1, 1>>>(d_a, d_b, d_out);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}
