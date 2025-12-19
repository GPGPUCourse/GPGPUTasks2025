#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"


using u32 = unsigned int;

__global__ void sparse_csr_matrix_vector_multiplication(
    const  u32* csr_row_offsets,
    const  u32* csr_columns,
    const  u32* csr_values,
    const  u32* vector_values,
    u32* output_buffer,
    const u32 nrows,
    const u32 nnz)
{
    const u32 row = blockIdx.x;

    if (row >= nrows) return;

    const u32 tid = threadIdx.x;
    const u32 row_begin = csr_row_offsets[row];
    const u32 row_end = (row + 1 < nrows) ? csr_row_offsets[row + 1] : nnz;

    uint32_t partial_sum = 0;

    for (uint32_t i = row_begin + tid; i < row_end; i += blockDim.x) {
        partial_sum += csr_values[i] * vector_values[csr_columns[i]];
    }

    atomicAdd(&output_buffer[row], partial_sum);
}

__global__ void fill_zeroes(u32* output_buffer, u32 n) {
    const u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    output_buffer[i] = 0;
}

namespace cuda {

void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    const  u32* csr_row_offsets,
    const  u32* csr_columns,
    const  u32* csr_values,
    const  u32* vector_values,
    u32* output_buffer,
    const u32 nrows,
    const u32 nnz
)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize ws{GROUP_SIZE, nrows};
    ::fill_zeroes<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>(output_buffer, nrows);
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets,
        csr_columns,
        csr_values,
        vector_values,
        output_buffer,
        nrows,
        nnz
    );
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
