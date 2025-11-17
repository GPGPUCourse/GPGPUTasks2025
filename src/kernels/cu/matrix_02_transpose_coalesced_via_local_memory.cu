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
    const uint index_x_from = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index_y_from = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_x_from >= w || index_y_from >= h)
        return;

    __shared__ float local_data[(GROUP_SIZE_X+1)*GROUP_SIZE_Y];

    const uint local_index_to = threadIdx.y * (GROUP_SIZE_X+1) + threadIdx.x;
    const uint index_from = index_y_from * w + index_x_from;
    local_data[local_index_to] = matrix[index_from];

    __syncthreads();

    const uint delta = threadIdx.y - threadIdx.x;
    const uint index_x_to = blockIdx.x * blockDim.x + threadIdx.y;
    const uint index_y_to = blockIdx.y * blockDim.y + threadIdx.x;
    const uint index_to = index_x_to * h + index_y_to;
    const uint local_index_from = threadIdx.x * (GROUP_SIZE_X+1) + threadIdx.y;

    transposed_matrix[index_to] = local_data[local_index_from];
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
