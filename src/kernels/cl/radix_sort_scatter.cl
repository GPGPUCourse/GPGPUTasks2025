#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scatter(
    __global const uint* input,
    __global const uint* prefix_sum,
    __global       uint* output,
    unsigned int n,
    unsigned int offset,
    unsigned int mask)
{
    const unsigned int i = get_global_id(0);

    if (i < n) {
        const unsigned int bits = (input[i] >> offset) & mask;
        const unsigned int j = prefix_sum[i + bits * n] - 1;
        output[j] = input[i];
    }
}