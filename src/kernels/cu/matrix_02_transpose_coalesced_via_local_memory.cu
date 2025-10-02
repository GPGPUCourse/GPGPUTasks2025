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
    __shared__ float mem[GROUP_SIZE_X][GROUP_SIZE_Y + 1];
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // w
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; // h

    mem[threadIdx.x][threadIdx.y] = matrix[x + y * w];

    __syncthreads();

    const unsigned int xt = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int yt = blockIdx.x * blockDim.x + threadIdx.y;

    transposed_matrix[xt + yt * h] = mem[threadIdx.y][threadIdx.x];
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
