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
    const uint gid = get_global_id(0);
    const uint lane = gid % GROUP_SIZE;
    const uint block = gid / GROUP_SIZE;

    const uint tile = GROUP_SIZE * 2u;
    const uint start = block * tile;
    if (start >= n)
        return;

    const uint offset = (block == 0u) ? 0u : block_prefix[block - 1u];

    const uint ai = start + lane;
    const uint bi = ai + GROUP_SIZE;
    if (ai < n)
        out[ai] += offset;
    if (bi < n)
        out[bi] += offset;
}
