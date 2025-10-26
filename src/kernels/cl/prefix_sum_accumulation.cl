#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_accumulation(
    __global const uint* sums,
    __global       uint* prefix_sums,
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int i = get_global_id(0);

    if (i < n && (i + 1 >> pow2 & 1)) {
        prefix_sums[i] += sums[(i + 1 >> pow2) - 1];
    }
}
