#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* data,
    __global       uint* out,
    unsigned int n,
    unsigned int filter,
    unsigned int p)
{
    uint i = get_global_id(0);

    if(i >= n) {
        return;
    }
    out[i] = ((data[i] >> p) & 1) == filter;
}
