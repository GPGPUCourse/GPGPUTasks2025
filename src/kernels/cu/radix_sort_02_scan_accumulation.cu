#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define uint unsigned int

__global__ void radix_sort_02_scan_accumulation(
    const uint* buffer_fenwick_gpu, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
          uint* prefix_sum_accum_gpu, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint x = idx + 1;
    uint sum = 0;
    for (; x > 0; x -= x & -x) {
        sum += buffer_fenwick_gpu[x - 1];
    }

    prefix_sum_accum_gpu[idx] = sum;
}

namespace cuda {
void radix_sort_02_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer_fenwick_gpu, gpu::gpu_mem_32u &prefix_sum_accum_gpu, uint n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer_fenwick_gpu.cuptr(), prefix_sum_accum_gpu.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda