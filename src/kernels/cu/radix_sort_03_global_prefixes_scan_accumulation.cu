#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* pow2_sum,
          unsigned int* prefix_sum_accum,
          unsigned int n,
          unsigned int pow2)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int left_boundary_of_accumulate = idx / (1ull << pow2);

    if (idx >= n || left_boundary_of_accumulate == 0)

        return;

    if (!(idx & (1ull << pow2)))

        return;

    prefix_sum_accum[idx] += pow2_sum[left_boundary_of_accumulate - 1];


    /*if (idx < number_of_groups)
    {
        unsigned int offset = 0;
        for (unsigned int shift = 0; shift < 5; shift++)
        {
            if (idx & (1u << shift))
            {
                const unsigned int add_value = glob_group_counts_partial_prefix_sums[offset + idx / (1u << shift)];
                atomicAdd(&global_group_prefixes[idx], add_value);
            }
            offset += number_of_groups / (1u << shift);
        }
    }*/
}

namespace cuda {
void radix_sort_03_prefix_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
