#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global uint* buffer2,
    unsigned int n)
{
    unsigned int new_n = (n + 1) / 2;

    unsigned int i = get_global_id(0);

    if (i >= new_n)
        return;

    if (2 * i + 1 < n)
        buffer2[i] = buffer1[2 * i] + buffer1[2 * i + 1];
    else
        buffer2[i] = buffer1[2 * i];
}
