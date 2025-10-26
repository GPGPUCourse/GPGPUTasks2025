#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    const unsigned int shift,
    const unsigned int n)
{
    __local unsigned int local_data[NUM_BOXES];
    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);
    if (local_idx < NUM_BOXES) {
        local_data[local_idx] = 0u;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        const unsigned int val = (buffer1[idx] >> shift) & RADIX_MASK; 
        atomic_inc(&local_data[val]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_idx < NUM_BOXES) {
        buffer2[idx / GROUP_SIZE * NUM_BOXES + local_idx] = local_data[local_idx];
    }
}
