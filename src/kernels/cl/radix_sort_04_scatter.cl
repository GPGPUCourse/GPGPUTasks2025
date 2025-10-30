#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in,
    __global const uint* prefix,
    __global const uint* group_offsets,
    __global uint* out,
    const uint pass_shift,
    const uint N)
{
    const uint i = get_global_id(0);
    if (i >= N) return;

    const uint gix = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);

    const uint v = in[i];
    const uint d = (v >> pass_shift) & MASK;

    __local uint digits[GROUP_SIZE];
    digits[lid] = d;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_rank = 0u;
    for (uint j = 0; j < lid; ++j) {
        uint idx = gix * lsz + j;
        if (idx < N && digits[j] == d) ++local_rank;
    }

    uint base = prefix[d] + group_offsets[gix * RADIX + d];
    uint pos  = base + local_rank;

    out[pos] = v;
}
