#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* pow2_sum,
          unsigned int* next_pow2_sum,
           unsigned int prev_size,
           unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)

        return;

    next_pow2_sum[idx] = (2 * idx + 1) < prev_size ? pow2_sum[2 * idx] + pow2_sum[2 * idx + 1] : pow2_sum[2 * idx];

   /* __shared__ unsigned int local_partial_sums[31];

    if (idx < 16) {
        local_partial_sums[idx] = global_group_counts[idx];
    } else {
        local_partial_sums[idx] = 0;
        }
        
    __syncthreads();

    if (idx < 8)
    {
        local_partial_sums[16 + idx] = local_partial_sums[idx * 2] + local_partial_sums[idx * 2 + 1];
    }

    __syncthreads();

    if (idx < 4)
    {
        local_partial_sums[24 + idx] = local_partial_sums[16 + idx * 2] + local_partial_sums[16 + idx * 2 + 1];
    }

    __syncthreads();

    if (idx < 2)
    {
        local_partial_sums[28 + idx] = local_partial_sums[24 + idx * 2] + local_partial_sums[24 + idx * 2 + 1];
    }

    __syncthreads();

    if (idx == 0)
    {
        local_partial_sums[30] = local_partial_sums[28] + local_partial_sums[29];
    }

    __syncthreads();

    if (idx < 31) {
        buffer2[idx] = local_partial_sums[idx];
    }*/


    // implicit sync by kernel end
}

namespace cuda {
void radix_sort_02_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
