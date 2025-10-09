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
    __shared__ float cache_matrix[GROUP_SIZE + GROUP_SIZE_Y];
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int local_i = threadIdx.x;
    const unsigned int local_j = threadIdx.y;

    float val = 0.0f;
    if (i < w && j < h) {
        val = matrix[j * w + i];
    }
    cache_matrix[(blockDim.x + 1) * local_j + local_i] = val;
    __syncthreads();

    const unsigned int i_start_t = blockIdx.y * blockDim.y;
    const unsigned int j_start_t = blockIdx.x * blockDim.x;
    if (i < w && j < h) {
        transposed_matrix[(j_start_t + local_j) * h
            + (i_start_t + local_i)] = cache_matrix[(blockDim.y + 1) * local_i + local_j];
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
