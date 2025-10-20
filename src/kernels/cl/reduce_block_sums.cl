#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// Compute per-block sums only (one uint per work-group)
// Mapping: one work-group processes 2*GROUP_SIZE elements
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
reduce_block_sums(
    __global const uint* in,
    __global uint* block_sums,
    unsigned int n)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0); // == GROUP_SIZE

    const uint tile = lsize * 2u;
    const uint start = gid * tile;
    if (start >= n)
        return;

    const uint ai = start + lid;
    const uint bi = ai + lsize;

    // Load with zero padding
    uint a = (ai < n) ? in[ai] : 0u;
    uint b = (bi < n) ? in[bi] : 0u;

    __local uint temp[2 * GROUP_SIZE];
    temp[lid] = a;
    temp[lid + lsize] = b;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Upsweep reduction to get tile total in temp[tile-1]
    for (uint offset = 1u; offset < tile; offset <<= 1u) {
        uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < tile)
            temp[idx] += temp[idx - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u)
        block_sums[gid] = temp[tile - 1u];
}

