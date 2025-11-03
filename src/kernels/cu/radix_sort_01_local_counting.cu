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
    unsigned int a1,
    unsigned int a2)
{
    __shared__ unsigned int local[2 * 1024];

    const unsigned int i = threadIdx.x;
    const unsigned int j = blockIdx.x * blockDim.x + i;

    if (i < 2 * blockDim.x)
        local[i] = 0;
    __syncthreads();

    if (j < a1) {
        unsigned int v = buffer1[j];
        unsigned int bit = (v >> a2) & 1;
        atomicAdd(&local[bit * blockDim.x + i], 1);
        buffer3[j] = bit;
        buffer2[j] = 1 - bit;
    }
    __syncthreads();
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, gpu::gpu_mem_32u& buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    size_t shared_mem_bytes = 2 * workSize.cuBlockSize().x * sizeof(unsigned int);

    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_bytes, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
