#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* in,
    __global uint* counts_local,
    const uint pass_shift,
    const uint N
)
{
    const uint gid  = get_global_id(0);
    const uint lid  = get_local_id(0);
    const uint lsz  = get_local_size(0);
    const uint gix  = get_group_id(0);

    __local uint hist[RADIX];

    for (uint d = lid; d < RADIX; d += lsz) {
        hist[d] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < N) {
        const uint v = in[gid];
        const uint d = (v >> pass_shift) & MASK;
        atomic_inc(&hist[d]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint d = lid; d < RADIX; d += lsz) {
        counts_local[gix * RADIX + d] = hist[d];
    }
}
