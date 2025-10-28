#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* prefix0,
    __global const uint* prefix1,
    __global const uint* data,
    __global       uint* next_data,
    unsigned int n,
    unsigned int offset,
    unsigned int p)
{
    uint i = get_global_id(0);

    if (i >= n) {
        return;
    }

    if(((data[i] >> p) & 1) == 0) {
        next_data[prefix0[i] - 1] = data[i];
    } else {
        next_data[offset + prefix1[i] - 1] = data[i];
    }
}