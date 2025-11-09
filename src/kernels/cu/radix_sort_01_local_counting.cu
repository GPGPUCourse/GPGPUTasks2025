#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int n,
    unsigned int bit)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int group_counts[RADIX];

    if (threadIdx.x < RADIX) {
        group_counts[threadIdx.x] = 0;
    }

    __syncthreads();

    if (index < n) {
        const unsigned int value = buffer1[index];
        const unsigned int digit = (value >> (RADIX_BITS * bit)) & RADIX_MASK;
        atomicAdd(&group_counts[digit], 1);
    }

    __syncthreads();

    if (threadIdx.x < RADIX) {
        unsigned int pos = (n + GROUP_SIZE - 1) / GROUP_SIZE * threadIdx.x + blockIdx.x;
        buffer2[pos] = group_counts[threadIdx.x];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
