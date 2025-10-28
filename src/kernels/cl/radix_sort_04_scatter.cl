#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_zeroes,
    __global       uint* output,
    unsigned int n,
    unsigned int prefix_offset,
    unsigned int bit)
{
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    uint value = input[i];
    uint cnt_z = prefix_zeroes[prefix_offset + i];

    if (value & (1u << bit)) {
        output[prefix_zeroes[prefix_offset + n - 1] + (i - cnt_z)] = value;
    } else {
        output[cnt_z - 1] = value;
    }
}