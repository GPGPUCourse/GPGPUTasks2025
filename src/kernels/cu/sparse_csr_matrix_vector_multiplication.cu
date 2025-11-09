#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
                const unsigned int* csr_row_offsets,
                const unsigned int* csr_columns,
                const unsigned int* csr_values,
                const unsigned int* vector_values,
                      unsigned int* output_vector_values,
                      unsigned int  nrows)
{
    const unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / SMALL_GROUP_SIZE;
    const unsigned int thr_id_in_warp = threadIdx.x % SMALL_GROUP_SIZE;
    const unsigned int warp_id_in_block = threadIdx.x / SMALL_GROUP_SIZE;
    
    if (warp_id >= nrows)
        return;
    
    const unsigned int row = warp_id;
    

    unsigned int row_start = csr_row_offsets[row];
    unsigned int row_end = csr_row_offsets[row + 1];
    unsigned int row_length = row_end - row_start;
    
    __shared__ unsigned int sums[SMALL_GROUP_SIZE / SMALL_GROUP_SIZE];

    if (thr_id_in_warp == 0) {
        sums[warp_id_in_block] = 0;
    }
    __syncthreads();


    for (unsigned int i = row_start + thr_id_in_warp; i < row_end; i += SMALL_GROUP_SIZE) {
        unsigned int col = csr_columns[i];
        atomicAdd(&sums[warp_id_in_block], csr_values[i] * vector_values[col]);
    }

    __syncthreads();
    
    if (thr_id_in_warp == 0) {
        output_vector_values[row] = sums[warp_id_in_block];
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& csr_row_offsets,
    const gpu::gpu_mem_32u& csr_columns,
    const gpu::gpu_mem_32u& csr_values,
    const gpu::gpu_mem_32u& vector_values,
    gpu::gpu_mem_32u& output_vector_values,
    unsigned int nrows)
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
        nrows
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
