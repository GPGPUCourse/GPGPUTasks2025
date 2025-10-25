#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void copy_array(
    __global const uint* source,
    __global       uint* target,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n)
        target[i] = source[i];
}
