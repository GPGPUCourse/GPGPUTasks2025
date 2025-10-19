#include <cstdio>
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
    const unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    const unsigned int index2 = index / 2;

    if (index2 < n) {
        unsigned int sum1 = pow2_sum[index];
        unsigned int sum2 = (index + 1 < n) ? pow2_sum[index + 1] : 0;
        next_pow2_sum[index2] = sum1 + sum2;
    }
}

namespace cuda {
void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize,
            const uint32_t* pow2_sum, uint32_t* next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum, next_pow2_sum, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
