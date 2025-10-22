#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // contains next_n values
    unsigned int n)
{
    unsigned int next_n = (n + 1) / 2;
    const unsigned int index = get_global_id(0);
    if (index >= next_n) return;
    next_pow2_sum[index] = pow2_sum[(index << 1)];
    if ((index << 1) | 1 < n) {
        next_pow2_sum[index] += pow2_sum[((index << 1) | 1)];
    }
}