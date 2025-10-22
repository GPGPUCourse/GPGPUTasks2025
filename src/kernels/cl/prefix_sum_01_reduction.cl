#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* output_per_elem_sums,
    __global uint* output_block_sums,
    unsigned int n)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int block_start_id = group_id * GROUP_SIZE;
    const unsigned int global_id = get_global_id(0);

    __local unsigned int local_data[GROUP_SIZE];

    if (global_id < n) {
        local_data[local_id] = input[global_id];
    } else {
        local_data[local_id] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 1u; i < GROUP_SIZE; i <<= 1u) {
        uint tmp = 0u;
        if (local_id >= i) tmp = local_data[local_id - i];
        barrier(CLK_LOCAL_MEM_FENCE);
        local_data[local_id] = local_data[local_id] + tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n) output_per_elem_sums[global_id] = local_data[local_id];

    if (local_id == GROUP_SIZE - 1u) {
        if (block_start_id < n) {
            output_block_sums[group_id] = local_data[(uint)(min((uint)GROUP_SIZE, n - block_start_id) - 1u)];
        } else {
            output_block_sums[group_id] = 0u;
        }
    }
}
