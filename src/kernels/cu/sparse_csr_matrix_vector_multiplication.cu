#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* csr_row_offsets,
    const unsigned int* csr_columns,
    const unsigned int* csr_values,
    const unsigned int* vector_values,
    unsigned int* reuslt,
    unsigned int n, unsigned int m) // TODO input/output buffers
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int accumulator = 0;

    unsigned int row_from = csr_row_offsets[row];
    unsigned int row_to = csr_row_offsets[row + 1];
    // printf("read [%d;%d)\n", row_from, row_to);
    for (unsigned int i = row_from; i < row_to; ++i) {
        unsigned int col = csr_columns[i];
        // printf("acc+= vals[%d] * vec[%d]\n", i, col);
        accumulator += csr_values[i] * vector_values[col];
    }
    reuslt[row] = accumulator;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u csr_row_offsets,
    const gpu::gpu_mem_32u csr_columns,
    const gpu::gpu_mem_32u csr_values,
    const gpu::gpu_mem_32u vector_values,
    gpu::gpu_mem_32u reuslt,
    unsigned int n, unsigned int m) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(),
        csr_columns.cuptr(),
        csr_values.cuptr(),
        vector_values.cuptr(),
        reuslt.cuptr(),
        n, m
        // input_buffer.cuptr(),
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
