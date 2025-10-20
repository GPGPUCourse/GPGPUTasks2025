#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// Add scanned block offsets to each element of a block
// - `block_prefix[gid]` is the prefix sum of all previous blocks (inclusive of previous block)
// - We add this offset to every element of block gid
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
add_block_offsets(
    __global const uint* block_prefix, // size = num_blocks
    __global uint* out, // scanned tiles from pass 1
    unsigned int n)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);

    const uint tile = lsize * 2u;
    const uint start = gid * tile;
    if (start >= n)
        return;

    // Offset to add: prefix of all previous blocks; broadcast once per WG
    __local uint shared_offset;
    if (lid == 0u)
        shared_offset = (gid == 0u) ? 0u : block_prefix[gid - 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    uint offset = shared_offset;

    uint ai = start + lid;
    uint bi = ai + lsize;
    if (ai < n)
        out[ai] += offset;
    if (bi < n)
        out[bi] += offset;
}
