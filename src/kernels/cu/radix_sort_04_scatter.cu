#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    const unsigned int* group_prefix,
          unsigned int* output,
    unsigned int n,
    unsigned int shift)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ unsigned int local_counts[GROUP_SIZE * 16];
    for (int k = 0; k < 16; ++k) {
        local_counts[threadIdx.x * 16 + k] = 0;
    }

    __syncthreads();

    if (i < n) {
        unsigned int digit = (input[i] >> shift) & 0b1111;
        local_counts[threadIdx.x * 16 + digit] = 1;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (unsigned int group_elem = 1; group_elem < blockDim.x; ++group_elem) {
            for (unsigned int dig = 0; dig < 16; ++dig) {
                local_counts[16 * group_elem + dig] += local_counts[16 * (group_elem - 1) + dig];
            }
        }
    }

    __syncthreads();

    unsigned int elems_before = 0;

    if (i < n) {
        unsigned int digit = (input[i] >> shift) & 0b1111;

        for (unsigned int k = 0; k < digit; ++k) {
            unsigned int last_group_id = (n - 1) / blockDim.x;
            elems_before += group_prefix[last_group_id * 16 + k];
        }

        unsigned int prev_group_count =
            blockIdx.x > 0 ? group_prefix[(blockIdx.x - 1) * 16 + digit] : 0;

        output[elems_before + prev_group_count + local_counts[threadIdx.x * 16 + digit] - 1] = input[i];
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
