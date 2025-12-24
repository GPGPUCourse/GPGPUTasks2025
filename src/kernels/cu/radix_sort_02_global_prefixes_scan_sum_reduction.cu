#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int group_count,
    unsigned int pow2)
{
    unsigned int group_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (group_id >= group_count) return;

    unsigned int group_start = group_id * 16;

    for (int k = 0; k < 16; ++k) {
        unsigned int digit_pos = group_start + k;

        if (group_id < pow2) {
            buffer2[digit_pos] = buffer1[digit_pos];
        } else {
            unsigned int complimentary_group_start = (group_id - pow2) * 16;
            unsigned int complimentary_group_digit = complimentary_group_start + k;
            buffer2[digit_pos] = buffer1[digit_pos] + buffer1[complimentary_group_digit];
        }
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
