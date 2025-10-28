#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in,
    __global uint* positions,
    __global const uint* group_offsets,
    __global uint* out,
    const uint pass_shift,
    const uint N
)
{
    const uint i = get_global_id(0);
    if (i >= N) return;

    const uint v = in[i];
    const uint d = (v >> pass_shift) & MASK;

    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    __local uint local_hist[RADIX];
    for (uint x = lid; x < RADIX; x += lsz) local_hist[x] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_rank = 0u;
    for (uint j = 0; j < lsz; ++j) {
        uint idx = get_group_id(0) * lsz + j;
        if (idx >= N) break;
        uint vv = in[idx];
        uint dd = (vv >> pass_shift) & MASK;
        if (j < lid) local_rank += (dd == d);
    }

    const uint gix = get_group_id(0);
    const uint pos = positions[d] + group_offsets[gix * RADIX + d] + local_rank;
    out[pos] = v;
}