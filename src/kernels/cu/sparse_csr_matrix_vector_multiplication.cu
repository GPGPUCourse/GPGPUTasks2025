#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    unsigned int* csr_row_offsets,
    unsigned int* csr_columns,
    unsigned int* csr_values,
    unsigned int* vector_values,
    unsigned int* output_vector_values,
    unsigned int nrows,
    unsigned int n) 
{
    const unsigned int row_index = blockIdx.x;
    const unsigned int idx = threadIdx.x;

    __shared__ unsigned int elems[GROUP_SIZE];
    elems[threadIdx.x] = 0;
    if (row_index < nrows) {
        if (row_index == nrows - 1 && csr_row_offsets[row_index] + idx < n) {
            elems[threadIdx.x] = csr_values[csr_row_offsets[row_index] + idx] * vector_values[csr_columns[csr_row_offsets[row_index] + idx]];
        } else if (csr_row_offsets[row_index] + idx < csr_row_offsets[row_index + 1]) {
            elems[threadIdx.x] = csr_values[csr_row_offsets[row_index] + idx] * vector_values[csr_columns[csr_row_offsets[row_index] + idx]];
        }
    }
    __syncthreads();

    if (row_index < nrows) {
        if (threadIdx.x == 0) {
            unsigned int sum = 0;   
            for (unsigned int i = 0; i < GROUP_SIZE; i++) {
                sum += elems[i];
            }
            output_vector_values[row_index] = sum;
        }
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, gpu::gpu_mem_32u csr_row_offsets, gpu::gpu_mem_32u csr_columns, gpu::gpu_mem_32u csr_values, gpu::gpu_mem_32u vector_values, gpu::gpu_mem_32u output_vector_values, unsigned int nrows, unsigned int n) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(),
        csr_columns.cuptr(),
        csr_values.cuptr(),
        vector_values.cuptr(),
        output_vector_values.cuptr(),
        nrows,
        n
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
