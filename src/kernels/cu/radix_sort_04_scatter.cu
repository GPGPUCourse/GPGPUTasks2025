#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    const unsigned int* prefix_sum,
          unsigned int* output,
    unsigned int n,
    unsigned int bits_offset)
{
    const unsigned int idx_local = threadIdx.x;
    const unsigned int idx_group = blockIdx.x;
    const unsigned int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int groups_count = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    __shared__ unsigned int bucket_ids[GROUP_SIZE];

    const unsigned int value = (idx_global < n) ? input[idx_global] : 0;
    const unsigned int bucket_id = (value >> bits_offset) & BUCKET_MASK;

    bucket_ids[idx_local] = bucket_id;

    __syncthreads();

    unsigned int local_offset = 0;
    for (unsigned int i = 0; i < idx_local; ++i) {
        if (bucket_ids[i] == bucket_id) {
            ++local_offset;
        }
    }

    const unsigned int global_bucket_id = groups_count * bucket_id + idx_group;
    const unsigned int global_offset = (global_bucket_id > 0) ? prefix_sum[global_bucket_id - 1] : 0;

    if (idx_global < n) {
        output[global_offset + local_offset] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &prefix_sum, gpu::gpu_mem_32u &output, unsigned int n, unsigned int bits_offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), prefix_sum.cuptr(), output.cuptr(), n, bits_offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
