#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* zeros_per_group,
    __global       uint* ones_per_group,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);

    uint flag = 0;
    if (gid < n) {
        flag = (input[gid] >> bit) & 1u;
    }

    __local uint local_data[GROUP_SIZE];
    local_data[lid] = (gid < n && flag == 0) ? 1u : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_data[lid] += local_data[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        uint zeros = local_data[0];
        uint group_start = group * GROUP_SIZE;
        uint group_size = (group_start + GROUP_SIZE <= n) ? GROUP_SIZE : (group_start < n ? n - group_start : 0);
        zeros_per_group[group] = zeros;
        ones_per_group[group] = group_size - zeros;
    }
}
