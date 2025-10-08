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
    __shared__ float tile[GROUP_SIZE_X][GROUP_SIZE_X + 1];
    unsigned int i = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    unsigned int j = blockIdx.y * GROUP_SIZE_X + threadIdx.y;
    if (i >= w)
        return;

    #pragma unroll
    for (unsigned int offset = 0; offset < GROUP_SIZE_X; offset += GROUP_SIZE_Y) {
        const unsigned int y = j + offset;
        if (y >= h)
            break;

        tile[threadIdx.y + offset][threadIdx.x] = matrix[y * w + i];
    }

    __syncthreads();

    i = blockIdx.y * GROUP_SIZE_X + threadIdx.x;
    j = blockIdx.x * GROUP_SIZE_X + threadIdx.y;
    if (i >= h)
        return;

    #pragma unroll
    for (unsigned int offset = 0; offset < GROUP_SIZE_X; offset += GROUP_SIZE_Y) {
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
