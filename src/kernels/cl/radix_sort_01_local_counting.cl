#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint *buffer1, // original array
    __global uint *buffer2,
    unsigned int n,
    unsigned int from)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    if (index < n)
    {
        const uint value = buffer1[index];
        buffer2[index] = ((value >> from) & 1);
    }
}
