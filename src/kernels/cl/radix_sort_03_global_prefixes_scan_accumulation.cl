#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    const uint n,
    const uint offset_src,
    const uint offset_dest,
    const uint pow2)
{
    const uint index = get_global_id(0);

    if (index >= n) {
        return;
    }

    if (index & (1 << pow2)) {
        prefix_sum_accum[offset_dest + index] += pow2_sum[offset_src + (index >> pow2) - 1];
    }
}
