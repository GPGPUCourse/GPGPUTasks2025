#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
          unsigned int* count_buffer,
    unsigned int n,
    unsigned int bits_offset)
{
    const unsigned int idx_local = threadIdx.x;
    const unsigned int idx_group = blockIdx.x;
    const unsigned int idx_global = idx_group * blockDim.x + idx_local;
    const unsigned int groups_count = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    __shared__ unsigned int local_counts[BUCKET_COUNT];

    if (idx_local < BUCKET_COUNT) {
        local_counts[idx_local] = 0;
    }

    __syncthreads();

    if (idx_global < n) {
        unsigned int idx_bucket = (input[idx_global] >> bits_offset) & BUCKET_MASK;
        atomicAdd(local_counts + idx_bucket, 1);
    }

    __syncthreads();

    if (idx_local < BUCKET_COUNT) {
        unsigned int buffer_idx = groups_count * idx_local + idx_group;
        count_buffer[buffer_idx] = local_counts[idx_local];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &count_buffer, unsigned int n, unsigned int bits_offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), count_buffer.cuptr(), n, bits_offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
