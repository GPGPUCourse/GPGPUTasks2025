#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    size_t index = get_global_id(0);
    if (index < n && (index & ((size_t)1 << pow2)) != 0) {
        prefix_sum_accum[index] += pow2_sum[(index >> pow2) - 1];
    }
}
