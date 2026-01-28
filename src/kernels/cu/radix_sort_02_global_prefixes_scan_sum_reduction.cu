#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* prev_prefix,
    unsigned int* next_prefix,
    unsigned int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    int acc = 0;
    acc += (index * 2 < n) ? prev_prefix[index * 2] : 0;
    acc += (index * 2 + 1 < n) ? prev_prefix[index * 2 + 1] : 0;

    if (index < (n + 1) / 2) {
        next_prefix[index] = acc;
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& prev_prefix, gpu::gpu_mem_32u& next_prefix, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        prev_prefix.cuptr(), next_prefix.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
