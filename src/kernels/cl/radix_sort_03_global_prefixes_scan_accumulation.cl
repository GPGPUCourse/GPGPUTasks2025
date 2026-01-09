#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* zerosInclusive,     // [groups]
    __global       uint* zeroGroupOffset,   // [groups]
    __global       uint* totalZerosBuf     // [1]
)
{
    const uint gid = get_global_id(0);
    const uint groups = get_num_groups(0);
    if (gid >= groups) return;

    zeroGroupOffset[gid] = (gid == 0) ? 0u : zerosInclusive[gid - 1];

    if (gid == groups - 1) {
        totalZerosBuf[0] = zerosInclusive[gid];
    }
}
