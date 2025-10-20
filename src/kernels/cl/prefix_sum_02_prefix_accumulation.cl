#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* t, 
    __global       uint* t2,
    unsigned int layer_size,
    unsigned int offset)
{
    uint id = get_global_id(0);
    uint i = id + offset;

    if (id >= layer_size) {
        return;
    }

    t2[i] = (layer_size > 2 ? t2[i >> 1] : 0) + ((id & 1) ? t[i - 1] : 0);
}
