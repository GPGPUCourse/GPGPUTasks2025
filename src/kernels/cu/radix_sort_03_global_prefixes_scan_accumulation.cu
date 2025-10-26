#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* pow2_sum,
    unsigned int* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx & (1 << pow2)) {
        const unsigned int pow2_idx = idx / (1 << pow2) - 1;
        prefix_sum_accum[idx] += pow2_sum[pow2_idx];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda