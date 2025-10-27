#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* counts,   // [numGroups * 2]: {zerosCount[g], onesCount[g]}
    unsigned int n,                // total element count
    unsigned int bit)              // bit index (0..31)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = gid * GROUP_SIZE + lid;

    __local uint l_zeros[GROUP_SIZE];

    uint valid = idx < n ? 1u : 0u;
    uint v = valid ? values[idx] : 0u;
    uint is_zero = valid ? (uint)(((v >> bit) & 1u) == 0u) : 0u;

    l_zeros[lid] = is_zero;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = GROUP_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_zeros[lid] += l_zeros[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        uint group_begin = gid * GROUP_SIZE;
        uint group_valid = (group_begin < n) ? min((uint)GROUP_SIZE, n - group_begin) : 0u;
        uint zeros = l_zeros[0];
        uint ones  = group_valid - zeros;
        counts[gid * 2 + 0] = zeros;
        counts[gid * 2 + 1] = ones;
    }
}
