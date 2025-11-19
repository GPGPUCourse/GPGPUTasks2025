#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    uint n,
    uint pow2)
{
    const uint ind = get_global_id(0);
    if (ind < n) {
        const uint bit_mask = (1u << pow2);
        if ((ind + 1) & bit_mask) {
            const uint largest_offset = ((ind + 1) >> pow2) - 1;
            prefix_sum_accum[ind] += pow2_sum[largest_offset];
        }
    }
}
