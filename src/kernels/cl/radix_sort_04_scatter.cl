#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sums,
    __global       uint* output,
    unsigned int n,
    unsigned int bit_start)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    __local uint local_counts[GROUP_SIZE][BITS_COUNT];
    for (unsigned int i = 0; i < BITS_COUNT; i++) {
        local_counts[local_index][i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        ++local_counts[local_index][(input[idx] >> bit_start) & (BITS_COUNT - 1)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int bit_type = local_index; bit_type < BITS_COUNT; bit_type += GROUP_SIZE) {
        for (unsigned int i = 1; i < GROUP_SIZE; i++) {
            local_counts[i][bit_type] += local_counts[i - 1][bit_type];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //if (local_index == 0 && group_id == 0) {
    //   printf("group_id=%u data: \n", group_id);
    //   for (unsigned int bit_type = 0; bit_type < BITS_COUNT; bit_type++) {
    //       for(unsigned int i = 0; i < GROUP_SIZE; i++) {
    //            printf("%u ", local_counts[i][bit_type]);
    //       }
    //   printf("\n");
    //   }
    //   printf("\n");
    //}

    unsigned int local_pos = 0;
    unsigned int global_pos = 0;
    if (idx < n) {
        unsigned int bucket = (input[idx] >> bit_start) & (BITS_COUNT - 1);
        local_pos = local_counts[local_index][bucket] - 1;
        if (bucket * num_groups + group_id > 0) {
            global_pos = prefix_sums[bucket * num_groups + group_id - 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //printf("value=%u group_id=%u loc_id=%u global_pos=%u local_pos=%u\n", input[idx], group_id, local_index, global_pos, local_pos);


    if (global_pos + local_pos < n && idx < n) {
        output[global_pos + local_pos] = input[idx];
    }
}