#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void copy_buffer(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* from,
    unsigned int* to,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    to[index] = from[index];
}

namespace cuda {
void copy_buffer(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& from, gpu::gpu_mem_32u& to, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::copy_buffer<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(from.cuptr(), to.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
