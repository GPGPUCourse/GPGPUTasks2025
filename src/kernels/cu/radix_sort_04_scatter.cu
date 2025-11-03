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
    const unsigned int* buffer0,
    unsigned int* buffer3,
    unsigned int n)
{
    // TODO
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    unsigned int j = buffer2[n - 1];
    unsigned int jj = (i == 0) ? buffer2[0] : buffer2[i] - buffer2[i - 1];

    if (jj > 0) {
        buffer3[buffer2[i] - 1] = buffer1[i];
    } else {
        buffer3[j + buffer0[i] - 1] = buffer1[i];
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u& buffer1, const gpu::gpu_mem_32u& buffer2, const gpu::gpu_mem_32u& buffer0, gpu::gpu_mem_32u& buffer3, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer0.cuptr(), buffer3.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
