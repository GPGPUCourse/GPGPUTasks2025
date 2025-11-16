#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* output,
    unsigned int n,
    unsigned int pow2)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    if (gid >= n) return;
    const uint factor = 1u << pow2;

    if (((gid + 1) & factor) != 0) {
        const uint pow2_gid = ((gid + 1u) >> pow2) - 1u;
        output[gid * 16 + lid] += pow2_sum[pow2_gid * 16 + lid];
    }

}
