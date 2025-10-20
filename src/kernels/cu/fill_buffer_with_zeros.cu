#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void fill_buffer_with_zeros(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* buffer,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) return;

    buffer[index] = 0;
}

namespace cuda {
void fill_buffer_with_zeros(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::fill_buffer_with_zeros<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
