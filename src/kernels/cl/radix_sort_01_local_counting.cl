#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* local_histograms,
    uint n,
    uint shift)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint local_size = get_local_size(0);
    uint num_groups = get_num_groups(0);

    __local uint histogram[256];

    if (local_id < 256) {
        histogram[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint elements_per_group = (n + num_groups - 1) / num_groups;
    uint start = group_id * elements_per_group;
    uint end = min(start + elements_per_group, n);

    for (uint i = start + local_id; i < end; i += local_size) {
        uint value = input[i];
        uint digit = (value >> shift) & 0xFF;
        atomic_inc(&histogram[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 256) {
        local_histograms[group_id * 256 + local_id] = histogram[local_id];
    }
}
