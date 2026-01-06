#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* arr,
    __global       uint* cnt_buffer,
    unsigned int size,
    unsigned int iteration)
{
    const uint global_index = get_global_id(0);
    const uint group_index = get_group_id(0);
    const uint local_index = get_local_id(0);
    const uint num_groups = get_num_groups(0);

    __local uint cnt[BUCKET_SIZE];

    if (local_index < BUCKET_SIZE) {
        cnt[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint bucket_index = 0;
    if (global_index < size) {
        bucket_index = (arr[global_index] >> (iteration * LOG2_BUCKET_SIZE)) & BUCKET_MASK;
        atomic_inc(&cnt[bucket_index]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < BUCKET_SIZE) {
        uint write_pos = num_groups * local_index + group_index;
        cnt_buffer[write_pos] = cnt[local_index];
    }
}
