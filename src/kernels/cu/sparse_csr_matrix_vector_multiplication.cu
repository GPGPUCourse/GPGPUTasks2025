#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_offsets,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* vec,
    unsigned int* result,
    unsigned int rows,
    unsigned int nnz)
{
    __shared__ unsigned int buffer[GROUP_SIZE];

    const int start_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int row_index = blockIdx.x;
    const int local_index = threadIdx.y;

    const unsigned int row_offset = row_offsets[row_index];
    const unsigned int row_end = (row_index + 1) < rows ? row_offsets[row_index + 1] : nnz;
    const unsigned int row_size = row_end - row_offset;

    unsigned int acc = 0;
    for (int k = 0; k * GROUP_SIZE < row_size; k++) {
        const int index = row_offset + start_index + GROUP_SIZE * k;
        if (index < row_end) {
            acc += values[index] * vec[columns[index]];
        }
    }
    buffer[local_index] = acc;

    __syncthreads();

    if (local_index == 0) {
        unsigned int acc = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            acc += buffer[i];
        }

        // because of 1 group for each row
        result[row_index] = acc;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_offsets,
    const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& vec,
    gpu::gpu_mem_32u& result,
    unsigned int rows,
    unsigned int nnz)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(),
        columns.cuptr(),
        values.cuptr(),
        vec.cuptr(),
        result.cuptr(),
        rows,
        nnz);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
