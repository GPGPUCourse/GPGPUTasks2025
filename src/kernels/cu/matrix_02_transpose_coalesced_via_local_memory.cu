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
    const unsigned int input_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int input_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int result_x_index = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int result_y_index = blockIdx.x * blockDim.x + threadIdx.y;

    const unsigned int local_x_index = threadIdx.x;
    const unsigned int local_y_index = threadIdx.y;

    __shared__ float local_matrix[GROUP_SIZE];

    if (input_x_index < w && input_y_index < h) {
        // `local_x_index` изменяется по `threadIdx.x`, значит доступ на чтение coalesced.
        //
        // `(local_x_index + local_y_index) % GROUP_SIZE_Y` отвечает за сдвиг зависящий
        // от строки для того, чтобы потоки одного варпа попадали в разные банки памяти.
        local_matrix[local_y_index * GROUP_SIZE_X + (local_x_index + local_y_index) % GROUP_SIZE_Y] = matrix[input_y_index * w + input_x_index];
    }

    __syncthreads();

    if (result_x_index < h && result_y_index < w) {
        // `local_x_index` изменяется по `threadIdx.x`, значит доступ на запись coalesced.
        //
        // `(local_y_index + local_x_index) % GROUP_SIZE_Y` отвечает за сдвиг зависящий
        // от строки для того, чтобы потоки одного варпа попадали в разные банки памяти.
        transposed_matrix[result_y_index * h + result_x_index] = local_matrix[local_x_index * GROUP_SIZE_X + (local_y_index + local_x_index) % GROUP_SIZE_Y];
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
