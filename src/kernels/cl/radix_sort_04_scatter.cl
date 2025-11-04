#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_data,
    __global const uint* scanned_buckets,
    __global       uint* output_data,
    uint n,
    uint offset)
{
    __local uint shared_digits[GROUP_SIZE];

    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    uint my_value = 0;
    uint my_digit = 0;
    if (global_id < n) {
        my_value = input_data[global_id];
        const uint digit_mask = (1u << RADIX_WIDTH) - 1;
        my_digit = (my_value >> offset) & digit_mask;
    }

    shared_digits[local_id] = my_digit;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_pos = 0;
    if (global_id < n) {
        for (uint i = 0; i < local_id; ++i) {
            if (shared_digits[i] == my_digit) {
                ++local_pos;
            }
        }
    }

    uint base_pos = 0;
    uint bucket_idx = my_digit * num_groups + group_id;
    if (global_id < n) {
        if (bucket_idx > 0) {
            base_pos = scanned_buckets[bucket_idx - 1];
        }
    }

    if (global_id < n) {
        output_data[base_pos + local_pos] = my_value;
    }
}