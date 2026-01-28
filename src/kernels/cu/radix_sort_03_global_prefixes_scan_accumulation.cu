#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* current_prefix,
    unsigned int* prefix_accumulator,
    unsigned int pow2,
    unsigned int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int tindex = index + 1;

    if (index < n) {
        if ((tindex & (1 << pow2)) != 0) {
            const int segment_index = (tindex >> pow2) - 1;
            prefix_accumulator[index] += current_prefix[segment_index];
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& current_prefix,
    gpu::gpu_mem_32u& prefix_accumulator,
    unsigned int pow2,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(),
        workSize.cuBlockSize(), 0, stream>>>(
        current_prefix.cuptr(), prefix_accumulator.cuptr(), pow2, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
