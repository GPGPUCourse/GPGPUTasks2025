#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void uniform_add(__global uint* __restrict out,
                          __global const uint* __restrict block_offsets_inclusive, // скан block_sums
                          const uint n)
{
    const uint WG = GROUP_SIZE;
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    uint offset = (gid == 0) ? 0u : block_offsets_inclusive[gid - 1];

    const uint base = (uint)(2 * WG) * gid;
    const uint i0 = base + 2*lid;
    const uint i1 = i0 + 1;

    if (i0 < n) out[i0] += offset;
    if (i1 < n) out[i1] += offset;
}
