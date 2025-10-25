#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* buffer1,
    const unsigned int* buffer2,
    const unsigned int* buffer3,
          unsigned int* buffer4,
    const unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) return;

    const unsigned int add = buffer2[n - 1];
    if ((index == 0 && buffer2[index] == 1) ||
        (index != 0 && buffer2[index] != buffer2[index - 1])) {
        buffer4[buffer2[index] - 1] = buffer1[index];
    } else {
        buffer4[add + buffer3[index] - 1] = buffer1[index];
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1,
    const gpu::gpu_mem_32u &buffer2, const gpu::gpu_mem_32u &buffer3, gpu::gpu_mem_32u &buffer4, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), buffer4.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
