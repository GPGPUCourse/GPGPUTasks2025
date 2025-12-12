#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "helpers/indexes.cuh"
#include  <stdio.h>

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float atile[GROUP_SIZE_X * GROUP_SIZE_Y];
    __shared__ float btile[GROUP_SIZE_X * GROUP_SIZE_Y];

    int a_index = threadIdx.y * GROUP_SIZE_X + threadIdx.x;
    int b_index = threadIdx.x * GROUP_SIZE_Y + threadIdx.y;

    float accumulator = 0;
    for (int i = 0; i < k / GROUP_SIZE_X; i++) {
        atile[a_index] = a[a_index + GROUP_SIZE_X * i];
        btile[b_index] = b[b_index + GROUP_SIZE_Y * i];
        __syncthreads();

        accumulator += atile[a_index] * btile[b_index];
        __syncthreads();
    }

    int gBlockIndex = blockIdx.y * w + blockIdx.x;
    c[gBlockIndex] = accumulator;
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
