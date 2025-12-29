#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* scratch,
    unsigned int bit,
    unsigned int n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint base = group * GROUP_SIZE;
    const bool in_range = gid < n;

    __local uint local_zero[GROUP_SIZE];
    uint flag = 0;
    if (in_range) {
        const uint v = values[gid];
        flag = ((v >> bit) & 1u) ^ 1u;
    }
    local_zero[lid] = flag;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint prefix = 0;
    for (uint i = 0; i < lid; ++i)
        prefix += local_zero[i];
    if (in_range)
        scratch[gid] = prefix;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint group_zero = 0;
    if (lid == 0) {
        for (uint i = 0; i < GROUP_SIZE; ++i)
            group_zero += local_zero[i];
        if (base < n) {
            const uint groups = get_num_groups(0);
            const uint group_counts_base = n;
            if (group < groups)
                scratch[group_counts_base + group] = group_zero;
        }
    }
}
