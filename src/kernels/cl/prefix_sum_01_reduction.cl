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
    const unsigned int half_n = (n + 1) / 2;
    const unsigned int i = get_global_id(0);

    if (i >= half_n)
        return;

    uint sum = 0;
    for (uint k = 0; k < 2; ++k) {
        if (2 * i + k < n) {
            sum += pow2_sum[2 * i + k];
        } else {
            sum += 0;
        }
    }

    next_pow2_sum[i] = sum; // next_pow2_sum[i] = pow2_sum[2 * i + 0] + pow2_sum[2 *i + 1];
}
