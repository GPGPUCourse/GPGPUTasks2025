#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1)
{
    const unsigned tid = threadIdx.x;

    if (blockIdx.x != 0)
        return;

    for (unsigned bin = tid; bin < RADIX_BINS; bin += blockDim.x) {
        unsigned acc = 0u;
        unsigned idx = bin;

#pragma unroll 1
        for (unsigned b = 0; b < a1; ++b, idx += RADIX_BINS) {
            const unsigned val = buffer1[idx];
            buffer2[idx] = acc;
            acc += val;
        }
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} 