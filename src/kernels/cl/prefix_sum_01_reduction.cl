#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* from,
    __global       uint* to,
    __global       uint* prefix,
    unsigned int k,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i > n) {
        return;
    }

    const unsigned int is_adding = (((i + 1) >> k) & 1);
    prefix[i] += is_adding * from[is_adding * (((i + 1) >> k) - 1)];
    to[i] = ((i << 1) + 1 < (n >> k) ? (from[i << 1] + from[(i << 1) + 1]) : 0);
}
