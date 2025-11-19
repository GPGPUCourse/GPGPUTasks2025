#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* reduction_buffer,
          unsigned int* out,
    unsigned int n)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > n)
        return;

    uint result = 0;
    uint offset = 0;
    uint size = n;
    while (true) {
        if (index % 2 == 0) {
            result += reduction_buffer[offset+index];
        }
        if (index == 0)
            break;
        offset += size;
        size /= 2;
        index = (index-1) / 2;
    }

    const uint index1 = blockIdx.x * blockDim.x + threadIdx.x;
    // reduction_buffer[index1] = result;
    out[index1] = result;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
                                                     const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
