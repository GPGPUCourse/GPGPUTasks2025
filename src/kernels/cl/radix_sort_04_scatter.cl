#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* map,
    __global       uint* pref,
    __global       uint* buffer2,
    unsigned int n)
{
    uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    if (map[index] == 1) {
        buffer2[pref[index] - 1] = buffer1[index];
    } else {
        buffer2[pref[n - 1] + index - pref[index]] = buffer1[index];
    }
}