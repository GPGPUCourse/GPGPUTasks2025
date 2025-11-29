#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    unsigned int idx = get_global_id(0);
    unsigned int old_idx = idx * 2;

    if (old_idx >= n) {
        return;
    }
    if (old_idx + 1 >= n) {
        next_pow2_sum[idx] = pow2_sum[old_idx];
        return;
    }
    next_pow2_sum[idx] = pow2_sum[old_idx] + pow2_sum[old_idx + 1];
}
