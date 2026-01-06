#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sum,
    __global       uint* output,
                   uint n,
                   uint offset)
{
    const uint global_i = get_global_id(0);
    const uint local_i = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    __local uint group_bucket_ids[GROUP_SIZE];

    const uint element = global_i < n ? input[global_i] : 0;
    const uint bucket_id = (element >> (offset * LOG2_BUCKET_SIZE)) & BUCKET_MASK;

    group_bucket_ids[local_i] = bucket_id;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_offset = 0;
    for (uint i = 0; i < local_i; ++i) {
        if (group_bucket_ids[i] == bucket_id) {
            ++local_offset;
        }
    }

    const uint global_bucket_id = num_groups * bucket_id + group_id;
    const uint global_offset = global_bucket_id > 0 ? prefix_sum[global_bucket_id - 1] : 0;

    if (global_i < n) {
        // printf("global offset %u local: %u elem: %d\n", global_offset, local_offset, element);
        output[global_offset + local_offset] = element;
    }
}