#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* prefix,
    __global const uint* data,
    __global       uint* next_data,
    unsigned int n)
{
    uint i = get_global_id(0);

    if (i >= n) {
        return;
    }
    uint offset = prefix[]

    if(data[i] == 0) {
        next_data[i] = data[prefix[i]];
    } else {
        next_data[offset + i] = data[prefix[i]];
    }
}