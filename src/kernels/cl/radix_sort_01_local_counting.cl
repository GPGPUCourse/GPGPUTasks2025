#include "helpers/rassert.cl"
#include "../defines.h"

#define BUCKET_MASK (BUCKET_COUNT - 1)
#define GET_BUCKET_IDX(value, bit_shift) (((value) >> (bit_shift)) & BUCKET_MASK)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input_data,
    __global       uint* buckets,
    uint n,
    uint offset)
{
    const uint BUCKET_COUNT = 1 << RADIX_WIDTH;
    __local uint workgroup_bucket[BUCKET_COUNT];

    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    if (local_id < BUCKET_COUNT) {
        workgroup_bucket[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        uint element = input_data[global_id];
        uint bucket_idx = GET_BUCKET_IDX(element, offset);
        atomic_inc(&workgroup_bucket[bucket_idx]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < BUCKET_COUNT) {
        uint write_pos = num_groups * local_id + group_id;
        buckets[write_pos] = workgroup_bucket[local_id];
    }
}