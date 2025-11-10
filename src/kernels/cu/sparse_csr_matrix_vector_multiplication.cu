#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(const unsigned int* row_offsets,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* vector_values,
    unsigned int* output,
    const unsigned int n)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    const unsigned int from = row_offsets[row];
    const unsigned int to = row_offsets[row + 1];

    unsigned long long sum = 0ULL;

    for (unsigned int id = from; id < to; ++id) {
        sum += (unsigned long long)values[id] * vector_values[columns[id]];
    }
    output[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_offset,
    const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& vector_values,
    const gpu::gpu_mem_32u& output,
    const unsigned int n) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offset.cuptr(),
        columns.cuptr(),
        values.cuptr(),
        vector_values.cuptr(),
        output.cuptr(),
        n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
