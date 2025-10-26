#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* dual,
    __global       uint* pref,
    unsigned int size,
    unsigned int step)
{
    const unsigned int index = get_global_id(0);
    const unsigned int next_index = index + 1;

    if (index >= size) {
        return;
    }

    if (((next_index >> step) & 1) == 1) {
        pref[index] += dual[(next_index >> step) - 1];
    }
}
