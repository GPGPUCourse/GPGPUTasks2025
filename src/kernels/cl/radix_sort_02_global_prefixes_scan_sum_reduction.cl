#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* num_count,
    __global uint* num_prefix,
    __global uint* bucket_size,
    unsigned int num_groups)
{
    const uint bucket = get_group_id(0);
    const uint local_index = get_local_id(0);

    if (bucket >= RAD_SIZE) {
        return;
    }

    __local uint s[GROUP_SIZE];
    __local uint tile_sum;
    uint global_sum = 0;

    for (uint offset = 0; offset < num_groups; offset += GROUP_SIZE) {
        const uint group_index = offset + local_index;

        if (group_index < num_groups) {
            s[local_index] = num_count[group_index * RAD_SIZE + bucket];
        } else {
            s[local_index] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint d = 1; d < GROUP_SIZE; d <<= 1) {
            uint idx = ((local_index + 1) * (d << 1)) - 1;
            if (idx < GROUP_SIZE) {
                s[idx] += s[idx - d];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_index == 0) {
            tile_sum = s[GROUP_SIZE - 1];
            s[GROUP_SIZE - 1] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint d = GROUP_SIZE >> 1; d > 0; d >>= 1) {
            uint idx = ((local_index + 1) * (d << 1)) - 1;
            if (idx < GROUP_SIZE) {
                uint t = s[idx - d];
                s[idx - d] = s[idx];
                s[idx] += t;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (group_index < num_groups) {
            num_prefix[group_index * RAD_SIZE + bucket] = s[local_index] + global_sum;
        }

        global_sum += tile_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        bucket_size[bucket] = global_sum;
    }
}
