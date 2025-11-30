#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_scatter(
    __global const uint* input,
    __global const uint* input_index,
    __global uint* output,
    __global uint* output_index,
    __global const uint* num_prefix,
    __global const uint* bucket_base,
    __global const uint* local_prefix,
    unsigned int n,
    unsigned int offset)
{
    const uint global_index = get_global_id(0);

    if (global_index >= n) {
        return;
    }

    const uint group_index = get_group_id(0);
    uint value = input[global_index];
    uint bucket = (value >> offset) & RAD_SIZE_MASK;
    uint local_shift = local_prefix[global_index];

    uint pos = bucket_base[bucket] + num_prefix[group_index * RAD_SIZE + bucket] + local_shift;
    output[pos] = value;
    output_index[pos] = input_index[global_index];
}
