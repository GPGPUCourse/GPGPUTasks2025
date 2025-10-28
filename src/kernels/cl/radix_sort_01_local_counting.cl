#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* in,
    __global uint* outs,
    __global uint* total_sums,
    uint n,
    uint bit_offset)
{
    // TODO maybe make GROUP_SIZE = BIT_GRANULARITY_EXP and instead make multiple iterations
    __local uint buffer[BIT_GRANULARITY_EXP];

    size_t global_id = get_global_id(0);
    size_t local_id = get_local_id(0);
    if (local_id < BIT_GRANULARITY_EXP) {
        buffer[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        uint val = in[global_id];
        val = (val >> bit_offset) & GRANULARITY_MASK;
        // printf("block %u local add to %u\n", global_id / GROUP_SIZE, val);
        atomic_add(&buffer[val], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < BIT_GRANULARITY_EXP) {
        // printf("for block %u got local_id %u with value %u \n", global_id / GROUP_SIZE, local_id, buffer[local_id]);
        outs[(global_id / GROUP_SIZE) * BIT_GRANULARITY_EXP + local_id] = buffer[local_id];
        atomic_add(total_sums + local_id, buffer[local_id]);
    }
}
