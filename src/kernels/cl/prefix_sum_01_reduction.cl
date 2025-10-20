#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n,
    unsigned int next_n)
{
    const unsigned int idx = get_global_id(0);
    if (idx >= next_n) return;
    
    const uint left_idx = idx << 1;
    const uint right_idx = left_idx | 1;
    
    uint result = pow2_sum[left_idx];
    if (right_idx < n) {
        result += pow2_sum[right_idx];
    }
    next_pow2_sum[idx] = result;
}
