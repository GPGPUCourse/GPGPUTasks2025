#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global uint* prefix_sum_accum, // целим в prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const uint idx = get_global_id(0);
    if (idx >= n)
        return;

    const uint factor = (1u << pow2);
    if ((idx + 1) & factor) {
        prefix_sum_accum[idx] += pow2_sum[((idx + 1) >> pow2) - 1];
    }
}
