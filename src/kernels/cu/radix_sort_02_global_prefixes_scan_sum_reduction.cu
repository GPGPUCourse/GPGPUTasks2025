#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_02_global_prefixes_scan_first_step(
    const unsigned int* buffer1,
    unsigned int* buffer2,
    unsigned int n,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int mask)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int fullMask = (1u << bitCount) - 1u;

    if (index < n) {
        unsigned int value = (buffer1[index] >> offset) & fullMask;
        if (value == mask) {
            buffer2[index] = 1;
        } else {
            buffer2[index] = 0;
        }
    }
}

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* buffer1,
    unsigned int* buffer2,
    unsigned int prev_n,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    const unsigned int index2 = index / 2;

    if (index2 < n) {
        unsigned int sum1 = buffer1[index];
        unsigned int sum2 = (index + 1 < prev_n) ? buffer1[index + 1] : 0;
        buffer2[index2] = sum1 + sum2;
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(
    const gpu::gpu_mem_32u& buffer1,
    gpu::gpu_mem_32u& buffer2,
    unsigned int n,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int mask)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    unsigned int currentOffset = 0;
    unsigned int prevOffset = 0;
    unsigned int prev_n = n;
    for (int i = 0; n >= 1; n = (n + 1) / 2, ++i) {
        gpu::WorkSize workSize(256, n);
        if (i == 0) {
            ::radix_sort_02_global_prefixes_scan_first_step<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), n, bitCount, offset, mask);
        } else {
            ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer2.cuptr() + prevOffset, buffer2.cuptr() + currentOffset, prev_n, n);
        }
        CUDA_CHECK_KERNEL(stream);
        prevOffset = currentOffset;
        currentOffset += n;
        prev_n = n;
        if (n == 1) {
            break;
        }
    }
}
} // namespace cuda
