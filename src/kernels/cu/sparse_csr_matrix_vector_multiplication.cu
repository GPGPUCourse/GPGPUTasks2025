#ifdef CLANGD
#include <__clang_cuda_builtin_vars.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    uint const* csr_row_offsets, // 0..nrow
    uint const* csr_columns, // 0..nnz
    uint const* csr_values, // 0..nnz
    uint const* vector_values, // 0..ncol
    uint      * output_vector_values, // 0..nrow
    uint nnz,
    uint nrows,
    uint ncols )
{
    // TODO
    const uint r = blockIdx.x * blockDim.x + threadIdx.x;

    const uint row_start = csr_row_offsets[r];
    const uint row_end = csr_row_offsets[r+1];

    uint acc = 0;
    for (uint i = row_start; i < row_end; i++) {
        uint mtrx_val = csr_values[i];
        uint vec_val = vector_values[csr_columns[i]];
        acc += mtrx_val * vec_val;
    }
    
    output_vector_values[r] = acc;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    gpu::WorkSize const& workSize,
    gpu::gpu_mem_32u const& csr_row_offsets_gpu,
    gpu::gpu_mem_32u const& csr_columns_gpu,
    gpu::gpu_mem_32u const& csr_values_gpu,
    gpu::gpu_mem_32u const& vector_values_gpu,
    gpu::gpu_mem_32u      & output_vector_values_gpu,
    uint nnz,
    uint nrows,
    uint ncols )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets_gpu.cuptr(),
        csr_columns_gpu.cuptr(),
        csr_values_gpu.cuptr(),
        vector_values_gpu.cuptr(),
        output_vector_values_gpu.cuptr(),
        nnz, nrows, ncols
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
