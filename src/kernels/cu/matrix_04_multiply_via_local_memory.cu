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
    const unsigned int row_c = blockIdx.y * GROUP_SIZE_X + threadIdx.y;
    const unsigned int col_c = blockIdx.x * GROUP_SIZE_X + threadIdx.x;

    __shared__ float tile_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __shared__ float tile_b[GROUP_SIZE_X][GROUP_SIZE_X];

    float acc = 0.;

    const unsigned int num_tiles = k / GROUP_SIZE_X;

    for (unsigned int tile_idx = 0; tile_idx < num_tiles; ++tile_idx)
    {
        const unsigned int col_a = tile_idx * GROUP_SIZE_X + threadIdx.x;
        const unsigned int row_b = tile_idx * GROUP_SIZE_X + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = a[row_c * k + col_a];
        tile_b[threadIdx.y][threadIdx.x] = b[row_b * w + col_c];

        __syncthreads();

#pragma unroll
        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i)
        {
            acc += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row_c < h && col_c < w)
        c[row_c * w + col_c] = acc;
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
