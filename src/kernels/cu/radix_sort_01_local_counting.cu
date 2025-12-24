#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
          unsigned int* group_counts,
    unsigned int n,
    unsigned int shift)
{
    if (threadIdx.x == 0) {
        unsigned int group_start = blockIdx.x * 16;

        for (int dig = 0; dig < 16; ++dig) {
            group_counts[group_start + dig] = 0;
        }

        for (unsigned int group_elem_id = 0; group_elem_id < blockDim.x; ++group_elem_id) {
            unsigned int i = group_elem_id + blockIdx.x * blockDim.x;

            if (i >= n) break;

            unsigned int digit = (input[i] >> shift) & 0b1111;
            group_counts[group_start + digit] += 1;
        }
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
