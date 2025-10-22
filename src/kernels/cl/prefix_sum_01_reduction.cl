#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    unsigned srcoff,
    __global uint* next_pow2_sum,
    unsigned dstoff,
    unsigned n)
{
    pow2_sum += srcoff;
    next_pow2_sum += dstoff;
    unsigned i = get_global_id(0);
    if(2 * i < n - 1) {
        next_pow2_sum[i] = pow2_sum[2 * i] + pow2_sum[2 * i + 1];
    } else if(2 * i == n - 1) {
        next_pow2_sum[i] = pow2_sum[2 * i];
    }
}
