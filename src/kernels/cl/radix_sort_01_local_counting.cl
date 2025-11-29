#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global uint* counts,
    unsigned int n,
    unsigned int bit)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = gid * 256 + lid;

    __local uint l_counts[16];

    if (lid < 16) {
        l_counts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint valid = idx < n ? 1u : 0u;
    if (valid) {
        uint v = values[idx];
        uint bucket = (v >> bit) & 0xF;
        atomic_add(&l_counts[bucket], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 16) {
        counts[gid * 16 + lid] = l_counts[lid];
    }
}
