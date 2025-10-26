#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* output,
    unsigned int n,
    unsigned int offset)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);

    __local int cache[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++) {
        cache[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int idx = (input[global_id] >> offset) & (BLOCK_SIZE - 1);
    atomic_add(&cache[idx], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < BLOCK_SIZE) {
        output[group_id * BLOCK_SIZE + local_id] = cache[local_id];
    }
}
