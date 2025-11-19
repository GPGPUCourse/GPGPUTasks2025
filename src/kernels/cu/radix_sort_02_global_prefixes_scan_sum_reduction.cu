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
    unsigned int* reduction_buffer,
    unsigned int n)
{
    constexpr uint iters = 8;
    static_assert(GROUP_SIZE == 1 << iters);

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i = 0; i < iters; i++) {
        if (index < n/2)
            reduction_buffer[n + index] = reduction_buffer[index * 2] + reduction_buffer[index * 2 + 1];
        reduction_buffer += n;
        n /= 2;
        index /= 2;
        __syncthreads();
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
                                                      gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer2.cuptr() + a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
