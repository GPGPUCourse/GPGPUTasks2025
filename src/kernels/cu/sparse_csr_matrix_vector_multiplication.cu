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
    unsigned int* output,
    const unsigned int nrows)
{
    const unsigned int row = blockIdx.x * 2 + (threadIdx.x >= 128);
    __shared__ unsigned int results[GROUP_SIZE];
    results[threadIdx.x] = 0;

    if (row < nrows) {
        const unsigned int offset = row_offsets[row];
        const unsigned int cols_cnt = row_offsets[row + 1] - offset;

        const unsigned int local_ind = threadIdx.x - (threadIdx.x >= 128 ? 128 : 0);
        if (local_ind < cols_cnt) {
            const unsigned int col_num = columns[offset + local_ind];
            results[threadIdx.x] = values[offset + local_ind] * vector[col_num];
        }

    }
    __syncthreads();

    if (row < nrows) {
        if (threadIdx.x == 0 || threadIdx.x == 128) {
            unsigned int sum = 0;
            for (unsigned int i = 0; i < GROUP_SIZE / 2; i++) {
                sum += results[i + threadIdx.x];
            }
            output[row] = sum;
        }
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u& row_offsets, const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values, const gpu::gpu_mem_32u& vector, gpu::gpu_mem_32u& output,
    const unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), columns.cuptr(), values.cuptr(),
        vector.cuptr(), output.cuptr(), nrows
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
