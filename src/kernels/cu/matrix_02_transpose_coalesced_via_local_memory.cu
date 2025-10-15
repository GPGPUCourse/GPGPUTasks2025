#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* matrix, // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    size_t block_x = blockIdx.x * blockDim.x;
    size_t block_y = blockIdx.y * blockDim.y;
    size_t i = threadIdx.x + block_x;
    size_t j = threadIdx.y + block_y;
    __shared__ float shmem[(GROUP_SIZE_X + 1) * GROUP_SIZE_Y];
    if (i < w && j < h) {
        shmem[threadIdx.x + threadIdx.y * (GROUP_SIZE_X + 1)] = matrix[j * w + i];
    }
    __syncthreads();
    if ((block_x + threadIdx.y) < w && (block_y + threadIdx.x) < h) {
        transposed_matrix[(block_x + threadIdx.y) * h + (block_y + threadIdx.x)] = shmem[threadIdx.y + threadIdx.x * (GROUP_SIZE_X + 1)];
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
