#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned int glob_id = get_global_id(0);
    if (glob_id >= (n + 1) / 2) {
        return;
    }
    uint first = pow2_sum[glob_id * 2];
    uint second = (n % 2 != 0 && glob_id == ((n + 1) / 2 - 1)) ? 0u : pow2_sum[glob_id * 2 + 1];

    next_pow2_sum[glob_id] = first + second;
}
