#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const uint x = get_global_id(0);
    if (x * 2 >= n) {
        return;
    }
    next_pow2_sum[x] = pow2_sum[x * 2] + (x * 2 + 1 < n ? pow2_sum[x * 2 + 1] : 0);

    // printf("x: %d   sum: %d   pow2_sum[%d]: %d   pow2_sum[%d]:%d\n", 
    //     x, next_pow2_sum[x], x * 2, pow2_sum[x * 2], x * 2 + 1, (x * 2 + 1 < n ? pow2_sum[x * 2 + 1] : 0));
}