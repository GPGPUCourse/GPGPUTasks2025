#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
        const unsigned int* row_offsets,
        const unsigned int* columns,
        const unsigned int* values,
        const unsigned int* vector,
        unsigned int* result
    )
{
    const unsigned int idx_block = blockIdx.x;
    const unsigned int idx_local = threadIdx.x;

    __shared__ unsigned int tile[GROUP_SIZE];

    tile[idx_local] = 0;

    const unsigned int offset_left = row_offsets[idx_block];
    const unsigned int offset_right = row_offsets[idx_block + 1];

    for (unsigned int t = offset_left + idx_local; t < offset_right; t += GROUP_SIZE) {
        tile[idx_local] += values[t] * vector[columns[t]];
    }

    __syncthreads();

    if (idx_local == 0) {
        result[idx_block] = 0;

        for (const unsigned int t : tile) {
            result[idx_block] += t;
        }
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u& row_offsets, const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values, const gpu::gpu_mem_32u& vector, gpu::gpu_mem_32u& result)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), columns.cuptr(), values.cuptr(), vector.cuptr(), result.cuptr()
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
