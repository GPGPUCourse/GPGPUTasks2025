#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* in,
    uint n,
    unsigned int level)
{
    __local uint data[GROUP_SIZE];
    size_t global_id = get_global_id(0);
    size_t local_idx = get_local_id(0);
    size_t block_idx = global_id / BIT_GRANULARITY_EXP;
    size_t pos_in_block = global_id % BIT_GRANULARITY_EXP;

    size_t initial_block_idx = (block_idx + 1) * level - 1;
    if (initial_block_idx < n) {
        data[local_idx] = in[initial_block_idx * BIT_GRANULARITY_EXP + pos_in_block];
    } else {
        data[local_idx] = 0;
    }
    size_t local_block_idx = local_idx / BIT_GRANULARITY_EXP;
    size_t pos_in_local_block = local_idx % BIT_GRANULARITY_EXP;
#pragma unroll
    for (int iter = 1; iter <= GROUP_SIZE_LOG - BIT_GRANULARITY; ++iter) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (((local_block_idx + 1) & ((1 << iter) - 1)) == 0) {
            data[local_idx] += data[(local_block_idx - (1 << (iter - 1))) * BIT_GRANULARITY_EXP + pos_in_local_block];
        }
    }
    // no need for barrier
    if (initial_block_idx < n) {
        in[initial_block_idx * BIT_GRANULARITY_EXP + pos_in_block] = data[local_idx];
    }
}
