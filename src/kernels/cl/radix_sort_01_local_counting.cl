#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input_data,
    __global       uint* histograms,
    uint n,
    uint offset)
{
    __local uint local_hist[NUM_DIGITS];

    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    if (local_id < NUM_DIGITS) {
        local_hist[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        uint digit = (input_data[global_id] >> offset) & RADIX_MASK;
        atomic_inc(&local_hist[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < NUM_DIGITS) {
        uint dst_index = num_groups * local_id + group_id;
        histograms[dst_index] = local_hist[local_id];
    }
}
