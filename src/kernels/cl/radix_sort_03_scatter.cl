#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_03_scatter(
    unsigned int n,
    __global const uint* buff1,
    __global uint* buff2,
    __global const uint* pref_sum,
    unsigned int bits,
    unsigned int digit,
    unsigned int offset,
    unsigned int clazz)
{
    const uint i = get_global_id(0);
    const uint mask = (1 << bits) - 1;

    if (i < n && ((buff1[i] >> (digit * bits)) & mask) == clazz) {
        buff2[offset + pref_sum[i] - 1] = buff1[i];
    }
}