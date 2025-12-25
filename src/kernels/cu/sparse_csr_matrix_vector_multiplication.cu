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
) {
    unsigned int target_row = blockIdx.x;

    // if (target_row >= nrows) {
    //     return;
    // }

    __shared__ unsigned int cache[GROUP_SIZE];

    cache[threadIdx.x] = 0;

    unsigned int start = row_offsets[target_row];
    unsigned int end = row_offsets[target_row + 1];
    
    unsigned int cur = start + threadIdx.x;
    while (cur < end) {
        cache[threadIdx.x] += values[cur] * vector[columns[cur]];
        cur += GROUP_SIZE;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        result[target_row] = 0;

        for (int i = 0; i < GROUP_SIZE; ++i) {
            result[target_row] += cache[i];
        }
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_offsets,
    const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& vector,
    gpu::gpu_mem_32u& result
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(),
        columns.cuptr(),
        values.cuptr(),
        vector.cuptr(),
        result.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
