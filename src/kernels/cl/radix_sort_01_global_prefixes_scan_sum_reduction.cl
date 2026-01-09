#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global uint* next_pow2_sum,
    unsigned int n)
{
    const uint i = get_global_id(0);
    if (i < n) {
        next_pow2_sum[i] = pow2_sum[2 * i] + pow2_sum[2 * i + 1];
    }
}