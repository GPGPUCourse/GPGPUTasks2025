#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    const unsigned int* per_chunk_counts_pow2_sum,      // contains chunks_count_pow2 * CASES_PER_PASS
          unsigned int*  per_chunk_counts_next_pow2_sum, // will contain chunks_count_next_pow2 * CASES_PER_PASS
    unsigned int chunks_count_pow2)
{
    const unsigned int chunks_count_next_pow2 = (chunks_count_pow2 + 1) / 2;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int chunk_id = i / CASES_PER_PASS;
    const unsigned int case_id = i % CASES_PER_PASS;

    if (chunk_id >= chunks_count_next_pow2)
        return;

    unsigned int sum = 0;
    for (unsigned int k = 0; k < 2; ++k) {
        if (2 * chunk_id + k < chunks_count_pow2) {
            sum += per_chunk_counts_pow2_sum[(2 * chunk_id + k) * CASES_PER_PASS + case_id];
        } else {
            sum += 0;
        }
    }

    per_chunk_counts_next_pow2_sum[chunk_id * CASES_PER_PASS + case_id] = sum; // next_pow2_sum[i] = pow2_sum[2 * i + 0] + pow2_sum[2 *i + 1];
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &per_chunk_counts_pow2_sum, gpu::gpu_mem_32u &per_chunk_counts_next_pow2_sum, unsigned int chunks_count_pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(per_chunk_counts_pow2_sum.cuptr(), per_chunk_counts_next_pow2_sum.cuptr(), chunks_count_pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
