#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    uint n)
{
    const uint index = get_global_id(0);
    if (index < (n + 1) / 2) {
        uint left = pow2_sum[2 * index];
        uint right = 0;
        if (2 * index + 1 < n) {
            right = pow2_sum[2 * index + 1];
        }
        next_pow2_sum[index] = left + right;
    }
}
