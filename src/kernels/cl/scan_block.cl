#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void scan_block(__global const uint* __restrict in,
                         __global       uint* __restrict out,
                         __global       uint* __restrict block_sums, // size = num_blocks
                         const uint n)
{
    const uint WG = GROUP_SIZE;
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);

    const uint base = (uint)(2 * WG) * gid;
    const uint i0 = base + 2*lid;
    const uint i1 = i0 + 1;

    __local uint temp[2 * GROUP_SIZE];

    uint x0 = (i0 < n) ? in[i0] : 0u;
    uint x1 = (i1 < n) ? in[i1] : 0u;

    temp[2*lid]   = x0;
    temp[2*lid+1] = x1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < 2*WG; offset <<= 1) {
        uint idx = (lid + 1) * (offset << 1) - 1;
        if (idx < 2*WG) {
            temp[idx] += temp[idx - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        block_sums[gid] = temp[2*WG - 1];
        temp[2*WG - 1]  = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = WG; offset > 0; offset >>= 1) {
        uint idx = (lid + 1) * (offset << 1) - 1;
        if (idx < 2*WG) {
            uint t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i0 < n) out[i0] = temp[2*lid] + x0;
    if (i1 < n) out[i1] = temp[2*lid+1] + x1;
}
