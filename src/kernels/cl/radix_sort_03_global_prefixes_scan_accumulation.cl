#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* counts_reduced,
    __global uint* prefix
)
{
    const uint lid = get_local_id(0);

    __local uint s[RADIX];

    uint x = counts_reduced[lid];
    s[lid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint off = 1; off < RADIX; off <<= 1) {
        uint t = s[lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= off) t += s[lid - off];
        barrier(CLK_LOCAL_MEM_FENCE);
        s[lid] = t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    prefix[lid] = s[lid] - x;
}
