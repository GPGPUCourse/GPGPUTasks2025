#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* base,
    __global const uint* pref,
    __global       uint* result,
    unsigned int size,
    unsigned int offset)
{
    const unsigned int index = get_global_id(0);

    if (index >= size || pref[index] == 0 || (index > 0 && pref[index] == pref[index - 1])) {
        return;
    }

    result[offset + pref[index] - 1] = base[index];
}
