#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* a,
    __global       uint* b,
    unsigned int offset,
    unsigned int n)
{
    const uint i = get_global_id(0);
    if (i >= n)
        return;
    uint value = a[i];
    if (i >= offset) {
        value += a[i - offset];
    }
    b[i] = value;
}
