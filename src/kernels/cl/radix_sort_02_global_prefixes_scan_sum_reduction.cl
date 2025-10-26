#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* base,
    __global       uint* dual,
    unsigned int size)
{
    const unsigned int index = get_global_id(0);

    if (index >= size / 2) {
        return;
    }

    dual[index] = base[index * 2] + base[index * 2 + 1];
}
