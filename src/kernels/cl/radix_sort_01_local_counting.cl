#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global uint* block_histograms,
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

    uint counts[RADIX_BUCKETS];
    for (uint digit = 0; digit < RADIX_BUCKETS; ++digit) {
        counts[digit] = 0u;
    }

    for (uint item = 0; item < ITEMS_PER_THREAD; ++item) {
        const uint idx = base_index + item * local_size + local_id;
        if (idx < n) {
            const uint value = values[idx];
            const uint digit = (value >> bit_offset) & (RADIX_BUCKETS - 1u);
            counts[digit] += 1u;
        }
    }

    __local uint shared[RADIX_BUCKETS * GROUP_SIZE];
    const uint shared_offset = local_id * RADIX_BUCKETS;
    for (uint digit = 0; digit < RADIX_BUCKETS; ++digit) {
        shared[shared_offset + digit] = counts[digit];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX_BUCKETS) {
        uint acc = 0u;
        for (uint offset = local_id; offset < RADIX_BUCKETS * local_size; offset += RADIX_BUCKETS) {
            acc += shared[offset];
        }
        block_histograms[group_id * RADIX_BUCKETS + local_id] = acc;
    }
}
