#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in_buffer,
    __global       uint* local_counting,
    __global       uint* prefix_sum_accum,
    __global       uint* out_buffer,
    unsigned int value,
    unsigned int n)
{
    const uint index = get_global_id(0);

    if (index >= n) return;

    const uint num_of_zeros = (value == 1) ? (n - prefix_sum_accum[n - 1]) : 0;
    const uint number = in_buffer[index];
    const uint pref_index = index > 0 ? prefix_sum_accum[index - 1] : 0;

    if (local_counting[index] == 1) out_buffer[num_of_zeros + pref_index] = number;
}