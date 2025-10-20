#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // contains n values
    unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int out_len = (n + 1) / 2;

    for (unsigned int i = id; i < out_len; i += stride) {
        unsigned int idx1 = 2 * i;
        unsigned int idx2 = idx1 + 1;
        unsigned int v1 = (idx1 < n) ? pow2_sum[idx1] : 0;
        unsigned int v2 = (idx2 < n) ? pow2_sum[idx2] : 0;
        next_pow2_sum[i] = v1 + v2;
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
