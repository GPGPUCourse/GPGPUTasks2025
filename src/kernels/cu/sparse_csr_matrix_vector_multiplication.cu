#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../../models.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const models::gpu::CSRMatrix matrix,
    const models::gpu::Array vector,
    uint* output
    )
{
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row + 1 >= matrix.offsets.length) return;

    const uint begin = matrix.offsets.array[row];
    const uint end   = matrix.offsets.array[row + 1];

    uint acc = 0;
    for (uint i = begin; i < end; ++i) {
        uint col = matrix.cols.array[i];
        acc += matrix.values.array[i] * vector.array[col];
    }

    output[row] = acc;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize, const models::CSRMatrix& matrix, const gpu::gpu_mem_32u& vector, gpu::gpu_mem_32u& output)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.to_gpu(), models::gpu::Array(vector), output.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda