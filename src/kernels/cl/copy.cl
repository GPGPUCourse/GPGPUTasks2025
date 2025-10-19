#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void copy(
    __global uint* src,
    __global uint* dst,
    unsigned int n)
{
    uint i = get_global_id(0);

    if (i < n)
        dst[i] = src[i];
}