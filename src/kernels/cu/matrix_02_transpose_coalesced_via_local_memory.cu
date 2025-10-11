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
    // TODO done
    __shared__ float local_data[GROUP_SIZE_X][GROUP_SIZE_X + 1];

    int x = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    int y = blockIdx.y * GROUP_SIZE_X + threadIdx.y;

    if (x < w && y < h) {
        local_data[threadIdx.y][threadIdx.x] = matrix[y * w + x];
    }

    __syncthreads();

    int tX = blockIdx.y * GROUP_SIZE_X + threadIdx.x;
    int tY = blockIdx.x * GROUP_SIZE_X + threadIdx.y;

    if (tX < h && tY < w) {
        transposed_matrix[tY * h + tX] = local_data[threadIdx.x][threadIdx.y];
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
