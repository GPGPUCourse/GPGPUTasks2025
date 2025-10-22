#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global uint* prefix_sums,
    __global const uint* block_sums,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index < n && index / GROUP_SIZE > 0) {
        // printf("block_add %d = %d %d\n", index, prefix_sums[index], block_sums[index / GROUP_SIZE - 1]);

        prefix_sums[index] += block_sums[index / GROUP_SIZE - 1];
    }
}
