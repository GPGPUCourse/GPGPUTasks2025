#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_map(
    __global uint* nums,
    __global uint* map_buf,
    unsigned int n,
    unsigned int bits_offset)
{
    uint global_i = get_global_id(0);
    uint i = global_i % n;
    uint global_i_digit = global_i / n;
    uint x = -1;
    if (i < n) x = nums[i];
    uint digit = (x >> bits_offset) & ((1 << DIGITS_SIZE) - 1);

    if (global_i < (n * (1<<DIGITS_SIZE)) && global_i_digit == digit)
        map_buf[global_i] = 1;
}