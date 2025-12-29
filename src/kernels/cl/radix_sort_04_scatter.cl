#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* scratch,
    __global       uint* output,
    unsigned int bit,
    unsigned int n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint base = group * GROUP_SIZE;
    if (gid >= n)
        return;

    const uint v = values[gid];
    const uint is_one = (v >> bit) & 1u;
    const uint zp = scratch[gid];
    const uint groups = get_num_groups(0);
    const uint group_counts_base = n;
    const uint group_prefix_base = n + groups;
    const uint group_zero_prefix = scratch[group_prefix_base + group];
    const uint total_zero = scratch[group_prefix_base + groups];
    const uint ones_pos = total_zero + gid - (group_zero_prefix + zp);
    const uint pos = (is_one == 0) ? (group_zero_prefix + zp) : ones_pos;

    output[pos] = v;
}
