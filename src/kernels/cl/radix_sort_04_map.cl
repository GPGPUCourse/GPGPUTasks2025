#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_map(
    unsigned int n,
    __global const uint* input,
    __global uint* output,
    unsigned int bits,
    unsigned int digit,
    unsigned int clazz)
{
    const uint i = get_global_id(0);
    const uint mask = (1 << bits) - 1;
    if (i < n) {
        output[i] = (((input[i] >> (digit * bits)) & mask) == clazz);
    }
}
