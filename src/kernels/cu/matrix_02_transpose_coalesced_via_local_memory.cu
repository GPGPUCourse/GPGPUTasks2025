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
    constexpr int SIZE_X = GROUP_SIZE_X; // + 1 does not give any advantage
    constexpr int SIZE_Y = GROUP_SIZE_Y;
    constexpr int SIZE = SIZE_X * SIZE_Y;

    __shared__ float buffer[SIZE];

    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int local_index = local_y * SIZE_X + local_x;
    const int local_transposed_index = local_x * SIZE_X + local_y;

    const int x = block_x + local_x;
    const int y = block_y + local_y;
    const int tx = block_y + local_x;
    const int ty = block_x + local_y;

    const int index = y * w + x;
    const int tindex = ty * h + tx;

    buffer[local_index] = (x < w && y < h) ? matrix[index] : 0;
    __syncthreads();

    if (tx < h && ty < w) {
        transposed_matrix[tindex] = buffer[local_transposed_index];
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
