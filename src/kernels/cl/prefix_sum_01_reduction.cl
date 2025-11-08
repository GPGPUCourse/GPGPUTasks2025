#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    int idx = get_global_id(0);
    if (2 * idx >= n) {
        return;
    } else {
        if (idx * 2 - 1 == n) {
            next_pow2_sum[idx] = pow2_sum[n - 1];
        } else {
            next_pow2_sum[idx] = pow2_sum[2 * idx] + pow2_sum[2 * idx + 1];
        }
    }
}
