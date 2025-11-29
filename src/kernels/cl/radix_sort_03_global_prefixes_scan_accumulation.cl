#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    unsigned int idx = get_global_id(0);

    if (idx >= n) {
        return;
    }

    unsigned int num_of_pow2 = (idx + 1) >> pow2;

    if ((num_of_pow2 & 1) == 1) {
        prefix_sum_accum[idx] += pow2_sum[num_of_pow2 - 1];
    }
}
