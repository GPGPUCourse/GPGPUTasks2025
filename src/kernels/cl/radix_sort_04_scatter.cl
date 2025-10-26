#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values_in,
    __global uint* values_out,
    __global const uint* block_prefixes,
    __global const uint* bucket_bases,
    unsigned int n,
    unsigned int bit_offset,
    unsigned int num_groups)
{
    const uint group_id = get_group_id(0);
    if (group_id >= num_groups) {
        return;
    }

    const uint local_id = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint elements_per_group = local_size * ITEMS_PER_THREAD;
    const uint base_index = group_id * elements_per_group;

    uint values_local[ITEMS_PER_THREAD];
    uint digits_local[ITEMS_PER_THREAD];
    uint valid_mask[ITEMS_PER_THREAD];

    for (uint item = 0; item < ITEMS_PER_THREAD; ++item) {
        const uint idx = base_index + item * local_size + local_id;
        if (idx < n) {
            const uint value = values_in[idx];
            const uint digit = (value >> bit_offset) & (RADIX_BUCKETS - 1u);
            values_local[item] = value;
            digits_local[item] = digit;
            valid_mask[item] = 1u;
        } else {
            values_local[item] = 0u;
            digits_local[item] = 0u;
            valid_mask[item] = 0u;
        }
    }

    __local uint shared_offsets[RADIX_BUCKETS * GROUP_SIZE];
    __local uint item_bases[RADIX_BUCKETS];
    __local uint item_totals[RADIX_BUCKETS];
    if (local_id < RADIX_BUCKETS) {
        item_bases[local_id] = 0u;
        item_totals[local_id] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint block_hist_offset = group_id * RADIX_BUCKETS;
    for (uint item = 0; item < ITEMS_PER_THREAD; ++item) {
        const uint is_valid = valid_mask[item];
        const uint element_digit = digits_local[item];

        for (uint digit = 0; digit < RADIX_BUCKETS; ++digit) {
            const uint flag = (is_valid && element_digit == digit) ? 1u : 0u;
            shared_offsets[digit * local_size + local_id] = flag;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < RADIX_BUCKETS) {
            const uint digit = local_id;
            uint accum = 0u;
            const uint row_start = digit * local_size;
            for (uint lane = 0; lane < local_size; ++lane) {
                const uint idx = row_start + lane;
                const uint count = shared_offsets[idx];
                shared_offsets[idx] = accum;
                accum += count;
            }
            item_totals[digit] = accum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (is_valid) {
            const uint digit = element_digit;
            const uint local_prefix = shared_offsets[digit * local_size + local_id];
            const uint offset_within_block = item_bases[digit] + local_prefix;
            const uint global_index =
                bucket_bases[digit] +
                block_prefixes[block_hist_offset + digit] +
                offset_within_block;

            values_out[global_index] = values_local[item];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < RADIX_BUCKETS) {
            item_bases[local_id] += item_totals[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
