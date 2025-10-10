#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__device__ __forceinline__ void set(float* a, int n, int m, int i, int j, float x)
{
    if (i < n && j < m) {
        a[i * m + j] = x;
    }
}

__device__ __forceinline__ float get(const float* a, int n, int m, int i, int j)
{
    return (i < n && j < m) ? a[i * m + j] : 0.0f;
}

__global__ void matrix_transpose_naive(
    const float* matrix, // h x w
    float* transposed_matrix, // w x h
    unsigned int w,
    unsigned int h)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 0..w
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 0..h

    float data = get(matrix, h, w, i, j);
    set(transposed_matrix, w, h, j, i, data);
}

namespace cuda {
void matrix_transpose_naive(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_naive<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
