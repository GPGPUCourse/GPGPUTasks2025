#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,          // [n]
    __global       uint* zeroPrefix,     // [n]  exclusive count of zeros before element
    __global       uint* zerosPerGroup,  // [groups]
    uint n,
    uint bit_number
)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint groups = get_num_groups(0);

    const uint isZero = (gid < n) ? (((input[gid] >> bit_number) & 1u) ^ 1u) : 0u;

    __local uint scan[GROUP_SIZE];
    scan[lid] = isZero;    
    barrier(CLK_LOCAL_MEM_FENCE);

    // inclusive scan isZero inside group
    for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        uint add = (lid >= offset) ? scan[lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // exclusive prefix for element
    if (gid < n) {
        zeroPrefix[gid] = scan[lid] - isZero;
    }

    if (lid == GROUP_SIZE - 1) {
        zerosPerGroup[group] = scan[lid];
    }
}
