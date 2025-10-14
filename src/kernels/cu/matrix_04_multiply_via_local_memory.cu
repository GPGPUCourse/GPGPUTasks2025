#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a,
                       const float* b,
                             float* c,
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const int TILE_DIM = GROUP_SIZE_X;

    __shared__ float tile_a[TILE_DIM][TILE_DIM];
    __shared__ float tile_b[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    float sum = 0.0f;

    for (int t = 0; t < k; t += TILE_DIM) {

        int a_load_row = row;
        int a_load_col = t + tx;

        int b_load_row = t + ty;
        int b_load_col = col;

        if (a_load_row < h && a_load_col < k) {
            tile_a[ty][tx] = a[a_load_row * k + a_load_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        if (b_load_row < k && b_load_col < w) {
            tile_b[ty][tx] = b[b_load_row * w + b_load_col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }

        __syncthreads();
    }

    if (col < w && row < h) {
        c[row * w + col] = sum;
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
