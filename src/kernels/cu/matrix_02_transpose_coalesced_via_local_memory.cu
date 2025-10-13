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

    
    const unsigned int loc_idx_x = threadIdx.x;
    const unsigned int col_idx = blockIdx.x * blockDim.x;
    const unsigned int idx_x = col_idx + loc_idx_x;

    const unsigned int loc_idx_y = threadIdx.y;
    const unsigned int row_idx = blockIdx.y * blockDim.y;
    const unsigned int idx_y = row_idx + loc_idx_y;

    __shared__ float local_data[GROUP_SIZE_Y * GROUP_SIZE_X];

    if (idx_x < w && idx_y < h) {
        local_data[loc_idx_y * GROUP_SIZE_X + loc_idx_x] = matrix[idx_y * w + idx_x];
    }

    __syncthreads();

    if (idx_x < w && idx_y < h) {
        transposed_matrix[(col_idx + loc_idx_y) * h + (row_idx + loc_idx_x)] = local_data[loc_idx_x * GROUP_SIZE_X + loc_idx_y];
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
