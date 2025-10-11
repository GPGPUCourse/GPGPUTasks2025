#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // TODO

    __shared__ float a_local[GROUP_SIZE_X][GROUP_SIZE_X];
    __shared__ float b_local[GROUP_SIZE_X][GROUP_SIZE_X];

    unsigned int x = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    unsigned int y = blockIdx.y * GROUP_SIZE_X + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; ++t) {
        if (x < w && t * GROUP_SIZE_X + threadIdx.x < k) {
            a_local[threadIdx.y][threadIdx.x] = a[y * k + t * GROUP_SIZE_X + threadIdx.x];
        } 

        if (y < h && t * GROUP_SIZE_X + threadIdx.y < k) {
            b_local[threadIdx.y][threadIdx.x] = b[(t * GROUP_SIZE_X + threadIdx.y) * w + x];
        } 

        __syncthreads();

        for (int k = 0; k < GROUP_SIZE_X; ++k) {
            sum += a_local[threadIdx.y][k] * b_local[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
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
