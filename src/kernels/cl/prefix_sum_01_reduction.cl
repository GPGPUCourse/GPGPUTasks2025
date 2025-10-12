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
    const unsigned int divided = (i + 1) >> k;
    const unsigned int is_adding = (divided & 1);
    prefix[i] = (k > 0) * prefix[i] + is_adding * from[is_adding * (divided - 1)];

    const unsigned int next_1 = i << 1;
    const unsigned int next_2 = next_1 + 1;
    const unsigned int size = n >> k;
    const unsigned int is_accumulating_2 = next_2 < size;
    const unsigned int is_accumulating_1 = next_1 < size;
    to[i] = (is_accumulating_1 * from[is_accumulating_1 * next_1]) +
            (is_accumulating_2 * from[is_accumulating_2 * next_2]);
}
