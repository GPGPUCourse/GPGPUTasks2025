#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
// #include <stdio.h>

__global__ void radix_sort_01_local_counting0(
    const unsigned int* input,
          unsigned int* counted,
          unsigned int n,
          unsigned int in_digit_idx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    // printf("i = %d, val = %d, counted = %d\n", i, input[i], !(input[i] & (1 << in_digit_idx)));
    counted[i] = !(input[i] & (1 << in_digit_idx));
}

__global__ void radix_sort_01_local_counting1(
    const unsigned int* input,
          unsigned int* counted,
          unsigned int n,
          unsigned int in_digit_idx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    // printf("i = %d, val = %d, counted = %d\n", i, input[i], !(input[i] & (1 << in_digit_idx)));
    counted[i] = (input[i] & (1 << in_digit_idx)) != 0;
}

namespace cuda {
// void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
//             const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
// {
//     gpu::Context context;
//     rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
//     cudaStream_t stream = context.cudaStream();
//     ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
//     CUDA_CHECK_KERNEL(stream);
// }

void RadixSortLocalCount0(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1,
    gpu::gpu_mem_32u &buffer2,
    unsigned int a1,
    unsigned int a2
){
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting0<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}

void RadixSortLocalCount1(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1,
    gpu::gpu_mem_32u &buffer2,
    unsigned int a1,
    unsigned int a2
){
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting1<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
