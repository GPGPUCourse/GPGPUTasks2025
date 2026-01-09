#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_data,
    __global const uint* prefix_sums,
    __global uint* output_data,
    unsigned int n,
    unsigned int bit_position,
    unsigned int count0)
{
    const uint global_id = get_global_id(0);
    
    if (global_id >= n) {
        return;
    }

    uint value = input_data[global_id];
    uint bit = (value >> bit_position) & 1;

    uint prefix_inv = prefix_sums[global_id];
    uint output_pos;

    if (bit == 0) {
        output_pos = prefix_inv - 1;
    } else {
        output_pos = count0 + global_id - prefix_inv;
    }
    
    output_data[output_pos] = value;
}
