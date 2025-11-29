#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_local_counting(
    __global const uint* data,
    __global uint* num_count,
    __global uint* local_prefix,
    unsigned int n,
    unsigned int offset)
{
    __local volatile uint local_count[RAD_SIZE];

    const uint global_group_index = get_group_id(0);
    const uint global_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint output_shift = global_group_index * RAD_SIZE;

    if (local_index < RAD_SIZE) {
        local_count[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_index < n) {
        const uint value = data[global_index];
        const uint bucket = (value >> offset) & RAD_SIZE_MASK;
        const uint rank = atomic_inc(&local_count[bucket]);

        local_prefix[global_index] = 0;
        uint base = global_group_index * GROUP_SIZE;
        for (uint i = 0; i < local_index; ++i) {
            uint idx = base + i;
            if (idx < n) {
                if (((data[idx] >> offset) & RAD_SIZE_MASK) == bucket) {
                    local_prefix[global_index]++;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < RAD_SIZE) {
        num_count[output_shift + local_index] = local_count[local_index];
    }
}
