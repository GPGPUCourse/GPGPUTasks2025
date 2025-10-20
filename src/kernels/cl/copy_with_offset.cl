#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void copy_with_offset(
    __global const uint* a,
    __global       uint* b, 
    unsigned int n,
    unsigned int offset_a,
    unsigned int offset_b)
{
    uint i = get_global_id(0);
    if (i < n) {
        b[offset_b + i] = a[offset_a + i];
    }
}
