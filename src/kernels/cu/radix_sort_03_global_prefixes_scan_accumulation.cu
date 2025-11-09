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
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int check_bit = 1u << pow2;
    if (idx < n && idx & check_bit) {
        unsigned int add_idx = (idx >> pow2) - 1;
        unsigned int add = pow2_sum[add_idx];
        prefix_sum_accum[idx] += add;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
