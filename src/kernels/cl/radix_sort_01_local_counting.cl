#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* buckets,
    unsigned int n,
    unsigned int bit_start)
{
    const unsigned int local_index = get_local_id(0);

    __local uint local_counts[GROUP_SIZE][BITS_COUNT];
    for (unsigned int i = 0; i < BITS_COUNT; i++) {
        local_counts[local_index][i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int idx = get_global_id(0);
    if (idx < n) {
        ++local_counts[local_index][(input[idx] >> bit_start) & (BITS_COUNT - 1)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int group = get_group_id(0);
    const unsigned int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    for (unsigned int bit_type = local_index; bit_type < BITS_COUNT; bit_type += GROUP_SIZE) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            sum += local_counts[i][bit_type];
        }
        buckets[bit_type * num_groups + group] = sum;
    }

}
