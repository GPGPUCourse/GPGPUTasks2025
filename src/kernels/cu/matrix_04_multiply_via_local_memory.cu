#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define N GROUP_SIZE_N

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

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int j = blockIdx.x * blockDim.x + x;
    int i = blockIdx.y * blockDim.y + y;
    float res = 0;
    for (int offset = 0; offset < k; offset += N) {
        __shared__ float aloc[N*N];
        __shared__ float bloc[N*N];
        aloc[y * N + x] = get(a, h, k, i, offset + x);
        bloc[y * N + x] = get(b, k, w, offset + y, j);
        __syncthreads();
        for (int n = 0; n < N; ++n) {
            res += aloc[y*N + n] * bloc[n * N + x];
        }
        __syncthreads();
    } 
    set(c, h, w, i, j, res);
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
