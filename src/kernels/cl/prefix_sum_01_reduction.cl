#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int m)
{
    const uint x = get_global_id(0);

    if ((x << 1) >= m){
        return;
    }
    next_pow2_sum[x] = pow2_sum[x << 1];
    if (((x << 1) | 1) < m) {
        next_pow2_sum[x] += pow2_sum[(x << 1) | 1];
    }
}
