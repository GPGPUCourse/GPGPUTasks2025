#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void prefix_sum_01_sum_reduction(
    const unsigned int* pow2_sum, // contains n values
    unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // trivial algorithm for now
    int acc = 0;
    acc += (index * 2 < n) ? pow2_sum[index * 2] : 0;
    acc += (index * 2 + 1 < n) ? pow2_sum[index * 2 + 1] : 0;

    if (index < (n + 1) / 2) {
        next_pow2_sum[index] = acc;
    }
}

namespace cuda {
void prefix_sum_01_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
