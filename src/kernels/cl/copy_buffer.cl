#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void copy_buffer(
    __global const uint* src,
    __global       uint* dest,
    const uint n,
    const uint offset_src,
    const uint offset_dest)
{
    const uint index = get_global_id(0);

    if (index >= n) {
        return;
    }

    dest[offset_dest + index] = src[offset_src + index];
}
