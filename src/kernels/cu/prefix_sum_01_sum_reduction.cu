#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_sum_reduction(
    unsigned int* reduction_buffer,
    unsigned int n)
{
    constexpr uint iters = 8;
    static_assert(GROUP_SIZE == 1 << iters);

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = 0; i < iters; i++) {
        if (index < n/2)
            reduction_buffer[n + index] = reduction_buffer[index * 2] + reduction_buffer[index * 2 + 1];
        reduction_buffer += n;
        n /= 2;
        index /= 2;
        __syncthreads();
    }
}

namespace cuda {
void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &reduction_buffer, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(reduction_buffer.cuptr() + offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
