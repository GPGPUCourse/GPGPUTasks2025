#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_sum_reduction(
    const unsigned int* pow2_sum, // contains n values
          unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;

    if ((2 * idx) < n) {
        sum += pow2_sum[(2 * idx)];
    }

    if ((2 * idx) + 1 < n) {
        sum += pow2_sum[(2 * idx) + 1];
    }

    next_pow2_sum[idx] = sum;
}

namespace cuda {
void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
