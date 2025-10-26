#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// Work-efficient Blelloch scan per block (tile = 2*GROUP_SIZE)
// - Outputs inclusive scan of input elements into `out`
// - Writes per-block sums into `block_sums` (one value per work-group)
// Mapping: one work-group scans 2*GROUP_SIZE elements
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
scan_block_inclusive(
    __global const uint* in,
    __global uint* out,
    __global uint* block_sums,
    unsigned int n)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);

    const uint tile = GROUP_SIZE * 2u;
    const uint start = gid * tile;
    if (start >= n)
        return;

    const uint ai = start + lid;
    const uint bi = ai + GROUP_SIZE;

    // Load with zero padding
    uint a = (ai < n) ? in[ai] : 0u;
    uint b = (bi < n) ? in[bi] : 0u;

    __local uint temp[2 * GROUP_SIZE];
    temp[lid] = a;
    temp[lid + GROUP_SIZE] = b;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Upsweep: after this, temp[tile-1] holds the tile total
    for (uint offset = 1u; offset < tile; offset <<= 1u) {
        uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < tile)
            temp[idx] += temp[idx - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Capture tile total before zeroing for downsweep
    uint tile_total = temp[tile - 1u];

    // Prepare for exclusive down-sweep
    if (lid == 0)
        temp[tile - 1u] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Downsweep: produces exclusive scan in temp
    for (uint offset = GROUP_SIZE; offset > 0u; offset >>= 1u) {
        uint idx = (lid + 1u) * (offset << 1u) - 1u;
        if (idx < tile) {
            uint t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Convert exclusive -> inclusive by adding original inputs (kept in registers)
    if (ai < n)
        out[ai] = temp[lid] + a;
    if (bi < n)
        out[bi] = temp[lid + GROUP_SIZE] + b;

    // Write per-block sum (sum of all real elements in the tile)
    if (lid == 0 && block_sums)
        block_sums[gid] = tile_total;
}
