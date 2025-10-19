#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_scatter(
    __global uint* in_buf,
    __global uint* map_buf,
    __global uint* prefix_sum_buf,
    __global uint* result_buf,
    unsigned int n)
{
    uint global_i = get_global_id(0);
    uint i = global_i % n;
    uint x = -1;
    uint mapped = 0;
    uint offset = 0;
    if (i < n) x = in_buf[i];
    if (global_i < n * (1 << DIGITS_SIZE)) mapped = map_buf[global_i];
    if (global_i < n * (1 << DIGITS_SIZE)) offset = prefix_sum_buf[global_i];

    if (i < n && mapped == 1) {
        result_buf[offset - 1] = x;
    }
}