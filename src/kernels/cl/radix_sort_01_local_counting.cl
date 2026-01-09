#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* in_buffer,
    __global       uint* out_buffer,
    unsigned int n,
    unsigned int offset)
{
    const uint index = get_global_id(0);

    if (index >= n) return;

    const uint number = in_buffer[index];
    out_buffer[index] = ((number >> offset) & 1u) == 0;
}

//const uint mask = (1u << offset_size) - 1;