#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "helpers/indexes.cuh"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    __shared__ float local_buffer[GROUP_SIZE_Y][GROUP_SIZE_X + 1]; // Is accessible for all threads in block

    int gx = global_index_axis_x();
    int gy = global_index_axis_y();

    int lx = thread_index_axis_x();
    int ly = thread_index_axis_y();


    local_buffer[ly][lx] = matrix[gy * w + gx];
    __syncthreads();

    int ox = work_group_index_axis_y() * GROUP_SIZE_Y + lx;
    int oy = work_group_index_axis_x() * GROUP_SIZE_X + ly;

    if (ox < h && oy < w) {
        transposed_matrix[oy * h + ox] = local_buffer[lx][ly];
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
