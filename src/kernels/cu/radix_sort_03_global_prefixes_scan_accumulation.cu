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
    const unsigned int* per_chunk_counts_pow2_sum, // contains chunks_count_pow2 * CASES_PER_PASS, pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
          unsigned int* per_chunk_counts_prefixes, // will contain chunks_count * CASES_PER_PASS prefix sums (inclusive)
    unsigned int chunks_count,
    unsigned int pow2)
{
//    const unsigned int local_id = threadIdx.x;
//    const unsigned int group_id = blockIdx.x;
    //const unsigned int chunks_count = (n + ELEMENTS_PER_CHUNK - 1) / ELEMENTS_PER_CHUNK; // div_ceil

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int chunk_id = i / CASES_PER_PASS;
    const unsigned int case_id = i % CASES_PER_PASS;

    const unsigned int period = (1 << (pow2 + 1));
    const unsigned int in_period_offset = ((1 << pow2) - 1);
    //const unsigned int index = (chunk_id / (1 << pow2)) * period + (in_period_offset + chunk_id % (1 << pow2));
    const unsigned int index = chunk_id;

    if (chunk_id >= chunks_count)
        return;

    unsigned int current_prefix_sum = 0;
    if (pow2 != 0) {
        current_prefix_sum = per_chunk_counts_prefixes[chunk_id * CASES_PER_PASS + case_id];
    }
//    if (chunk_id == 0 && case_id == 0){
//        printf("TEST456 chunks_count=%d\n", chunks_count);
//    }
//    if (chunk_id == 1 && case_id == 0 && k == 1)
//        printf("TEST143 chunk_id=%d case_id=%d current_prefix_sum=%d+%d k=%d decomp=%d index+1=%d (1 << k)=%d\n",
//            chunk_id, case_id, current_prefix_sum, per_chunk_counts_pow2_sum[(index_decomposition_future_sum / (1 << pow2)) * CASES_PER_PASS + case_id],
//            k, index_decomposition_future_sum, index + 1, (1 << k));

    // let's say (index+1) = 2^k1 + 2^k2 + 2^k3 + ..., where is k1 > k2 > k3 > ...
    bool fast_finish = true;
    unsigned int index_decomposition_future_sum = 0; // those 2^k1 + 2^k2 + ... , who already will be taken into account in future (with larger pow2)
    for (int k = 31; k >= 0; --k) {
        if ((index + 1) & (1 << k)) { // if we have 2^k in (index+1) decomposition
            if (pow2 == k) { // if currently in pow2_sum partial sums including part of our index decomposition
                curassert(index_decomposition_future_sum % (1 << pow2) == 0, 65754325);
                current_prefix_sum += per_chunk_counts_pow2_sum[(index_decomposition_future_sum / (1 << pow2)) * CASES_PER_PASS + case_id];
            }
            index_decomposition_future_sum += (1 << k);
        }
        if (fast_finish && pow2 == k) {
            break;
        }
    }
    if (!fast_finish) {
        curassert((index + 1) == index_decomposition_future_sum, 142341231);
    }

    per_chunk_counts_prefixes[chunk_id * CASES_PER_PASS + case_id] = current_prefix_sum;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &per_chunk_counts_pow2_sum, gpu::gpu_mem_32u &per_chunk_counts_prefixes, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(per_chunk_counts_pow2_sum.cuptr(), per_chunk_counts_prefixes.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
