#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input_data,
    __global uint* bits_inverted,
    unsigned int n,
    unsigned int bit_position)
{
    const uint global_id = get_global_id(0);

    if (global_id >= n) {
        return;
    }

    uint value = input_data[global_id];
    uint bit = (value >> bit_position) & 1;

    bits_inverted[global_id] = 1 - bit;
}
