#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    uint n,
    uint pow2)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    const uint index1 = index + 1;
    if (((index1 >> pow2) & 1) != 0) {
        prefix_sum_accum[index] += pow2_sum[(index1 >> pow2) - 1];
    }
}
