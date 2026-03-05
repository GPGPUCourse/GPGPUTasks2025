#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_group_sum,
    __global       uint* output,
    const uint n,
    const uint num_groups,
    const uint pow)
{
    const uint index = get_global_id(0);
    const uint group_index = get_group_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_group_sum[1 << RADIX_BITS][GROUP_SIZE];

    if (index < n) {
        for (int i = 0; i < (1 << RADIX_BITS); ++i) {
            local_group_sum[i][local_index] = (((input[index] >> pow) & ((1 << RADIX_BITS) - 1)) == i);
        }
    } else {
        for (int i = 0; i < (1 << RADIX_BITS); ++i) {
            local_group_sum[i][local_index] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < (1 << RADIX_BITS); ++i) {
        for (uint d = 1; d < GROUP_SIZE; d <<= 1) {
            uint s = (local_index >= d ? local_group_sum[i][local_index - d] : 0);
            barrier(CLK_LOCAL_MEM_FENCE);
            local_group_sum[i][local_index] += s;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index >= n) {
        return;
    }

    uint category = (input[index] >> pow) & ((1 << RADIX_BITS) - 1);
    uint base_offset = 0;
    for (uint i = 0; i < category; ++i) {
        base_offset += prefix_group_sum[i * num_groups + num_groups - 1];
    }
    uint category_offset = (group_index > 0 ? prefix_group_sum[category * num_groups + group_index - 1] : 0);
    uint local_offset = local_group_sum[category][local_index];
    output[base_offset + category_offset + local_offset - 1] = input[index];
}
