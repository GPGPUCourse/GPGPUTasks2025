#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains 2n values
    __global uint* next_pow2_sum, // will contain n values
    unsigned int n)
{
    const uint idx = get_global_id(0);
    if (idx >= n) {
        return;
    }
    const uint i = idx * 2;
    next_pow2_sum[idx] = pow2_sum[i] + pow2_sum[i + 1];
}
