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
    unsigned int a1,
    unsigned int a2)
{
    const unsigned int tid   = threadIdx.x;
    const unsigned int bid   = blockIdx.x;
    const unsigned int bins  = (1u << RADIX_BITS);
    const unsigned int mask  = bins - 1u;
    const unsigned int chunk = (a1 + gridDim.x - 1u) / gridDim.x;
    const unsigned int begin = bid * chunk;
    const unsigned int end   = min(a1, begin + chunk);

    __shared__ unsigned int sh_hist[bins];
    if (tid < bins) sh_hist[tid] = 0u;
    __syncthreads();

    unsigned int local_cnt[RADIX_BINS];
#pragma unroll
    for (unsigned int b = 0; b < bins; ++b) local_cnt[b] = 0u;

    for (unsigned int i = begin + tid; i < end; i += blockDim.x) {
        const unsigned int key = buffer1[i];
        const unsigned int bin = (key >> a2) & mask;
        ++local_cnt[bin];
    }

#pragma unroll
    for (unsigned int b = 0; b < bins; ++b) {
        const unsigned int v = local_cnt[b];
        if (v) atomicAdd(&sh_hist[b], v);
    }
    __syncthreads();

    if (tid < bins) {
        buffer2[bid * bins + tid] = sh_hist[tid];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda