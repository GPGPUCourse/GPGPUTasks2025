#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void
radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint *prefix,
    __global uint *ones,
    unsigned int n)
{
    uint x = n;
    uint sum = 0;
    while (x > 0)
    {
        sum += prefix[x - 1];
        x = (x & (x - 1));
    }
    *ones = sum;
}
