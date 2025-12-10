#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const uint* row_offset,
    const uint* column_ind,
    const uint* value,
    const uint* vector,
          uint* out,
          int n
) {
    const uint index = blockIdx.x;
    const uint thread_id = threadIdx.x;

    if (index > n) return;

    uint start = row_offset[index];
    uint end = row_offset[index+1];
    uint result = 0;

    for (; start < end; start += SMALL_GROUP_SIZE) {
        uint ind = start + thread_id;
        if (ind < end)
            result += value[ind] * vector[column_ind[ind]];
    }

    __syncthreads();
    __shared__ uint data[SMALL_GROUP_SIZE];
    data[thread_id] = result;
    __syncthreads();

    if (thread_id == 0) {
        uint result = 0;
        for (uint i = 0; i < SMALL_GROUP_SIZE; i++)
            result += data[i];
        out[index] = result;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b,
            const gpu::gpu_mem_32u &c, const gpu::gpu_mem_32u &d, gpu::gpu_mem_32u &e, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
       a.cuptr(), b.cuptr(), c.cuptr(), d.cuptr(), e.cuptr(), n
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
