#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const size_t gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    const size_t elems_per_block = 2 * GROUP_SIZE;
    const size_t block_id = gid / elems_per_block;

    if (block_id == 0) {
        return;
    }

    prefix_sum_accum[gid] += pow2_sum[block_id - 1];
}
