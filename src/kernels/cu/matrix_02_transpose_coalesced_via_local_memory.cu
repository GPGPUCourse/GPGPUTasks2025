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
    __shared__ float tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    int x_in = blockIdx.x * blockDim.x + threadIdx.x;
    int y_in = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_in < w && y_in < h) {
        tile[threadIdx.y][threadIdx.x] = matrix[y_in * w + x_in];
    }

    __syncthreads();

    int x_out = blockIdx.y * blockDim.y + threadIdx.x;
    int y_out = blockIdx.x * blockDim.x + threadIdx.y;

    if (x_out < h && y_out < w) {
        transposed_matrix[y_out * h + x_out] = tile[threadIdx.x][threadIdx.y];
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
