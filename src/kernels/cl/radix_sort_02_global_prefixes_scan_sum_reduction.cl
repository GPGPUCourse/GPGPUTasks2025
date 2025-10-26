#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* digits,
    __global       uint* pref_sum,
    uint n,
    uint distance)
{
    uint global_id = get_global_id(0);
    if (global_id < n) {
        if (global_id < distance) {
            pref_sum[global_id] = digits[global_id];    
            return;
        }

        pref_sum[global_id] = digits[global_id] + digits[global_id - distance];
    }
}
