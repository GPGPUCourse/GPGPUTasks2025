#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* in,
    __global const uint* reduced_pref_sums,
    __global const uint* total_sums,
    __global uint* out,
    uint n,
    uint bit_offset)
{
    __local uint buffer[GROUP_SIZE];
    size_t global_id = get_global_id(0);
    size_t local_id = get_local_id(0);
    uint actual_num = 0;
    if (global_id < n) {
        actual_num = in[global_id];
        buffer[local_id] = (actual_num >> bit_offset) & GRANULARITY_MASK;
    } else {
        buffer[local_id] = GRANULARITY_MASK;
    }

    // get prefix of our subnumber
    size_t block_idx = global_id / GROUP_SIZE; // sum is right-exclusive, thats why no '+1'
    uint pref = 0;
    while (block_idx > 0) {
        pref += reduced_pref_sums[BIT_GRANULARITY_EXP * (block_idx - 1) + buffer[local_id]];
        block_idx -= block_idx & -block_idx;
    }
    uint save_global_pref = pref;

    // get total count of lower subnumbers
    for (uint i = 0; i < buffer[local_id]; ++i) {
        pref += total_sums[i];
    }
    uint save_total_sums_pref = pref;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 0; i < local_id; ++i) {
        pref += buffer[i] == buffer[local_id];
    }
    // printf("global_id %u got pref %u and value %u: global_pref - %u; total_sums_pref - %u\n", global_id, pref, actual_num, save_global_pref, save_total_sums_pref);
    if (pref < n) {
        out[pref] = actual_num;
    }
}
