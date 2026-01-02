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
    const unsigned int local_x = threadIdx.x;
    const unsigned int local_y = threadIdx.y;

    const unsigned int global_col = blockIdx.x * blockDim.x + local_x;
    const unsigned int global_row = blockIdx.y * blockDim.y + local_y;

    __shared__ float tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (global_row < h && global_col < w) {
        tile[local_y][local_x] = matrix[global_row * w + global_col];
    }

    __syncthreads();

    const unsigned int transposed_col = blockIdx.y * blockDim.y + local_x;
    const unsigned int transposed_row = blockIdx.x * blockDim.x + local_y;

    if (transposed_row < w && transposed_col < h) {
        transposed_matrix[transposed_row * h + transposed_col] = tile[local_x][local_y];
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
