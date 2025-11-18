#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global uint* next_pow2_sum, // will contain ceil(n/2) values
    unsigned int n)
{
    uint m = (n + 1) / 2;
    uint idx = get_global_id(0);

    if (idx >= m)
        return;

    if (2 * idx + 1 < n)
        next_pow2_sum[idx] = pow2_sum[2 * idx] + pow2_sum[2 * idx + 1];
    else
        next_pow2_sum[idx] = pow2_sum[2 * idx];
}
