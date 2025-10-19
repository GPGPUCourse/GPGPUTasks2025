#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_03_add_block_sum(
    __global int* data,
    __global int* block_sums,
    unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    if (gid >= n) return;

    int offset = (group_id == 0) ? 0 : block_sums[group_id - 1];
    data[gid] += offset;
}
