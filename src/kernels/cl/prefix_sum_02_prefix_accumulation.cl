#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    int pow2)
{
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int pos_idx = i + 1;
    if ((pos_idx >> pow2) % 2 == 1) {
        prefix_sum_accum[i] += pow2_sum[pos_idx / (1 << pow2) - 1];
    } else {
        prefix_sum_accum[i] = prefix_sum_accum[i];
    }
}
