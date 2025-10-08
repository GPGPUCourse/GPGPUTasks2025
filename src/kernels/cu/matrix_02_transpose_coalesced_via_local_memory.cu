#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    constexpr unsigned int TILE_DIM = 32, BLOCK_ROWS = 8;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int j = blockIdx.y * TILE_DIM + threadIdx.y;

    if (i < w) {
        #pragma unroll
        for (unsigned int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
            const unsigned int y = j + offset;
            if (y >= h)
                break;

            tile[threadIdx.y + offset][threadIdx.x] = matrix[y * w + i];
        }
    }

    __syncthreads();

    if (i >= w)
        return;
    i = blockIdx.y * TILE_DIM + threadIdx.x;
    if (i >= h)
        return;
    j = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (unsigned int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const unsigned int y = j + offset;
        if (y >= w)
            return;

        transposed_matrix[y * h + i] = tile[threadIdx.x][threadIdx.y + offset];
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
