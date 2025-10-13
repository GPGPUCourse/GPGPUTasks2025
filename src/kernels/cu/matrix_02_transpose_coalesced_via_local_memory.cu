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
    unsigned global_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned local_x = threadIdx.x;
    unsigned local_y = threadIdx.y;
    unsigned group_x = blockIdx.x;
    unsigned group_y = blockIdx.y;
    __shared__ float buffer[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    if (global_x < w && global_y < h) {
        buffer[local_y][local_x] = matrix[global_y * w + global_x];
    }
    else {
        buffer[local_y][local_x] = 0.0f;
    }
    __syncthreads();
    unsigned transposed_x = group_y * GROUP_SIZE_Y + local_y;
    unsigned transposed_y = group_x * GROUP_SIZE_X + local_x;
    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = buffer[local_y][local_x];
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
