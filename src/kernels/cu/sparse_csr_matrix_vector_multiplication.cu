#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_ptr,
    const unsigned int* col_idx,
    const unsigned int* values,
    const unsigned int* x,
    unsigned int* y,
    unsigned int nrows,
    unsigned int ncols)

{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    unsigned long long acc = 0ull;
    const unsigned int start = row_ptr[row];
    const unsigned int end = row_ptr[row + 1];
    for (unsigned int k = start; k < end; ++k)
        acc += (unsigned long long)values[k] * x[col_idx[k]];
    y[row] = (unsigned int)acc;
}
namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_ptr,
    const gpu::gpu_mem_32u &col_idx,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &x,
    gpu::gpu_mem_32u &y,
    unsigned int nrows,
    unsigned int ncols)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_ptr.cuptr(), 
        col_idx.cuptr(), 
        values.cuptr(),
        x.cuptr(), 
        y.cuptr(), 
        nrows, ncols);


    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
