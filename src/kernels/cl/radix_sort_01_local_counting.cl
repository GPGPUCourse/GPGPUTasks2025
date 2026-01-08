#include "helpers/rassert.cl"
#include "../defines.h"

#define BUCKET_COUNT (1u << RADIX_WIDTH)
#define DIGIT(value, shift) (((value) >> (shift)) & (BUCKET_COUNT - 1))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* src,
    uint count,
    uint bit_shift,
    __global       uint* hist_out)
{
    __local uint hist_local[BUCKET_COUNT];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint grp = get_group_id(0);
    const uint groups_total = get_num_groups(0);

    for (uint b = lid; b < BUCKET_COUNT; b += GROUP_SIZE) {
        hist_local[b] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < count) {
        uint val = src[gid];
        uint bucket = DIGIT(val, bit_shift);
        atomic_inc(&hist_local[bucket]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = lid; b < BUCKET_COUNT; b += GROUP_SIZE) {
        uint dst_index = b * groups_total + grp;
        hist_out[dst_index] = hist_local[b];
    }
}
