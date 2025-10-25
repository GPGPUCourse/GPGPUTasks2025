#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
          unsigned int* buffer2,
          unsigned int* buffer3,
    const unsigned int n,
    const unsigned int k)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) return;

    buffer2[index] = 1 - ((buffer1[index] >> k) & 1);
    buffer3[index] = (buffer1[index] >> k) & 1;
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2,
            gpu::gpu_mem_32u &buffer3, unsigned int n, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), n, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
