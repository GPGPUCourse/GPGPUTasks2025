#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* pow2_sum,
    unsigned int* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int mask_shift = pow2;
    unsigned int mask = 1 << mask_shift;

    for (unsigned int i = id; i < n; i += stride) {
        if (((i + 1) & mask) > 0) {
            unsigned int index = (i + 1) >> mask_shift;
            if (index > 0)
                index -= 1;
            if (index < n) {
                prefix_sum_accum[i] += pow2_sum[index];
            }
        }
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum,
    gpu::gpu_mem_32u& prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
