#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../../models.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    uint* values,
    uint* cols,
    uint* offsets,
    uint* vector,
    uint* output,
    uint nrows
    )
{
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    const uint begin = offsets[row];
    const uint end   = offsets[row + 1];

    uint acc = 0;
    for (uint i = begin; i < end; ++i) {
        const uint col = cols[i];
        acc += values[i] * vector[col];
    }

    output[row] = acc;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& values, const gpu::gpu_mem_32u& cols, const gpu::gpu_mem_32u& offsets, const gpu::gpu_mem_32u& vector, gpu::gpu_mem_32u& output)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), cols.cuptr(), offsets.cuptr(), vector.cuptr(), output.cuptr(), offsets.number() - 1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda