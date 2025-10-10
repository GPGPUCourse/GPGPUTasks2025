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
    const unsigned int li = threadIdx.x;
    const unsigned int lj = threadIdx.y;

    const unsigned int gi = blockIdx.x * blockDim.x;
    const unsigned int gj = blockIdx.y * blockDim.y;

    const unsigned int i = gi + li;
    const unsigned int j = gj + lj;

    __shared__ float local_data[GROUP_SIZE_Y * GROUP_SIZE_X];

    if (i < w && j < h) {
        local_data[lj * GROUP_SIZE_X + li] = matrix[j * w + i];
    }

    __syncthreads();

    if (i < w && j < h) {
        transposed_matrix[(gi + lj) * h + (gj + li)] = local_data[li * GROUP_SIZE_X + lj];
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
