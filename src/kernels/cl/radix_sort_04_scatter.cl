#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* src,
    uint total_count,
    uint bit_shift,
    __global const uint* prefix_table,
    __global       uint* dst)
{
    __local uint digits[GROUP_SIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint grp = get_group_id(0);
    const uint groups_total = get_num_groups(0);

    uint value = 0;
    uint digit = 0;
    if (gid < total_count) {
        value = src[gid];
        digit = (value >> bit_shift) & (RADIX_BUCKET_COUNT - 1);
    }

    digits[lid] = digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint rank_in_group = 0;
    if (gid < total_count) {
        for (uint i = 0; i < lid; ++i) {
            rank_in_group += (digits[i] == digit);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < total_count) {
        uint bucket_index = digit * groups_total + grp;
        uint base_offset = (bucket_index > 0) ? prefix_table[bucket_index - 1] : 0;
        dst[base_offset + rank_in_group] = value;
    }
}
