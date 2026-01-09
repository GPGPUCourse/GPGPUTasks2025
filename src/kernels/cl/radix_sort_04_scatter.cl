#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in_buffer,
    __global       uint* prefix_sum_accum,
    __global       uint* out_buffer,
    unsigned int offset,
    unsigned int n)
{
    const uint index = get_global_id(0);

    if (index >= n) return;
    
    const uint number = in_buffer[index];
    const bool is_zero = ((number >> offset) & 1u) == 0;
    const uint zero_count = is_zero ? 0 : prefix_sum_accum[n - 1];
    const uint pref_index = is_zero ? prefix_sum_accum[index] - 1 : index - prefix_sum_accum[index];

    out_buffer[zero_count + pref_index] = number;
}