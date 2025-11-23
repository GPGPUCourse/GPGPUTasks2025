#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* vector_values,
    const unsigned int* row_offsets,
    const unsigned int n,
    const unsigned int rows,
    unsigned int* output_values
)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
        return;

    int l = -1, r = rows;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (row_offsets[m] <= index)
            l = m;
        else
            r = m;
    }

    const unsigned int row = l;
    const unsigned int offset = row_offsets[row];
    const unsigned int value = values[index];
    const unsigned int column = columns[index];

    atomicAdd(&output_values[row], value * vector_values[column]);
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &vector_values,
    const gpu::gpu_mem_32u& row_offsets,
    const unsigned int n,
    const unsigned int rows,
    gpu::gpu_mem_32u& output_values
)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        columns.cuptr(),
        values.cuptr(),
        vector_values.cuptr(),
        row_offsets.cuptr(),
        n,
        rows,
        output_values.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
