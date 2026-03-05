#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* group_sum,
    const uint n,
    const uint num_groups,
    const uint pow)
{
    const uint index = get_global_id(0);
    const uint group_index = get_group_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_group_sum[1 << RADIX_BITS][GROUP_SIZE];

    if (index < n) {
        for (int i = 0; i < (1 << RADIX_BITS); ++i) {
            local_group_sum[i][local_index] = (((input[index] >> pow) & ((1 << RADIX_BITS) - 1)) == i);
        }
    } else {
        for (int i = 0; i < (1 << RADIX_BITS); ++i) {
            local_group_sum[i][local_index] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint k = GROUP_SIZE / 2;
    while (k >= 32) {
        if (local_index < k) {
            for (int i = 0; i < (1 << RADIX_BITS); ++i) {
                local_group_sum[i][local_index] += local_group_sum[i][local_index + k];
            }
        }
        k /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    while (k >= 1) {
        if (local_index < k) {
            for (int i = 0; i < (1 << RADIX_BITS); ++i) {
                local_group_sum[i][local_index] += local_group_sum[i][local_index + k];
            }
        }
        k /= 2;
    }

    if (local_index == 0) {
        for (int i = 0; i < (1 << RADIX_BITS); ++i) {
            group_sum[i * num_groups + group_index] = local_group_sum[i][0];
        }
    }
}
