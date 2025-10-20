#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const unsigned int i = get_global_id(0);

    if (i >= (n + 1)/2) {
        return;
    }

    int sum = pow2_sum[i * 2];
    if (i * 2 + 1 < n) {
        sum += pow2_sum[i * 2 + 1];
    }
    next_pow2_sum[i] = sum;
}
