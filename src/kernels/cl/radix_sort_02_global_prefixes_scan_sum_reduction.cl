#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* cnt,
    __global       uint* prefix,
    __global       uint* blocks,
    const unsigned int cnt_size,
    unsigned int cur_bucket)
{
    __local uint sum[GROUP_SIZE];

    const uint group_id = get_group_id(0);
    const uint local_index = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint index = group_id * local_size + local_index;

    const uint base = cur_bucket * cnt_size;

    uint val = 0u;
    if (index < cnt_size) {
        val = cnt[base + index];
    }
    sum[local_index] = val;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = 1; j < local_size; j <<= 1) {
        uint to_add = 0u;
        if (local_index >= j) {
            to_add = sum[local_index - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        sum[local_index] += to_add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index + 1 == local_size) {
        blocks[cur_bucket * get_num_groups(0) + group_id] = sum[local_index];
    }
    if (index < cnt_size) {
        prefix[base + index] = sum[local_index] - val; 
    }
}
