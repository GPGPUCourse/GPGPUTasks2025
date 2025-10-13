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
    const unsigned int glob_col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int glob_row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int glob_index = glob_row * w + glob_col;

    __shared__ float workgroup_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (glob_col < w && glob_row < h)
    {
        workgroup_data[threadIdx.y][threadIdx.x] = matrix[glob_index];
    } else {
        workgroup_data[threadIdx.y][threadIdx.x] = 0.;
    }

    __syncthreads();

    //
    const unsigned int transposed_x = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int transposed_y = blockIdx.x * blockDim.x + threadIdx.y;

    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = workgroup_data[threadIdx.x][threadIdx.y];
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
