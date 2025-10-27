#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* pref_sum,
            __global uint* output,
    unsigned int bit,
    unsigned int n)
{
    const unsigned int index = get_global_id(0);
    if (index >= n) {
        return;
    }
    unsigned int output_index;
    if ((input[index] >> bit) & 1) {
        output_index = pref_sum[n - 1] + (index - pref_sum[index]);
    } else {
        output_index = pref_sum[index] - 1;
    }
    output[output_index] = input[index];
}