#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const uint glob_id = get_global_id(0);
    if (glob_id >= n)
        return;
    
    const uint bit = 1u << pow2;

    if (((glob_id + 1) & bit) != 0u) {
        const uint block_idx = (glob_id + 1) >> (pow2 + 1);
        const uint left_block = block_idx << 1;
        const uint carry = pow2_sum[left_block];
        prefix_sum_accum[glob_id] += carry;
    }
}
