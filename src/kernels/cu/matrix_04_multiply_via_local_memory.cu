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
    // GSX SHOULD BE EQUAL TO GSY
    unsigned constexpr TILE = GROUP_SIZE_X;

    __shared__ float lhs[TILE][TILE];
    __shared__ float rhs[TILE][TILE];

    float val = 0;

    for (unsigned int iter = 0; iter < (k + TILE - 1) / TILE; ++iter) {
        // Load tiles
        unsigned int lhs_row = (TILE * blockIdx.y + threadIdx.y);
        unsigned int lhs_col = iter * TILE + threadIdx.x;
        // if lhs_row, col ...
        lhs[threadIdx.y][threadIdx.x] = a[lhs_row * k + lhs_col];

        unsigned int rhs_row = (TILE * iter + threadIdx.y);
        unsigned int rhs_col = blockIdx.x * TILE + threadIdx.x;
        // if rhs_row, col ...
        rhs[threadIdx.y][threadIdx.x] = b[rhs_row * w + rhs_col];

        __syncthreads();

        // perform multiplication and accumulate the result
        for (unsigned int k_tile = 0; k_tile < TILE; ++k_tile) {
            val += lhs[threadIdx.y][k_tile] * rhs[k_tile][threadIdx.x];
        }

        __syncthreads();
    }
    
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    c[y * w + x] = val;
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
