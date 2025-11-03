#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   unsigned int  group_size,
                   unsigned int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int block_index = i / group_size;
    const unsigned int block_left_border = block_index * group_size;
    const bool is_left = block_index % 2 == 0;

    unsigned int corresponding_block_left_border;

    if (is_left) {
        corresponding_block_left_border = (block_index + 1) * group_size;
    } else {
        corresponding_block_left_border = (block_index - 1) * group_size;
    }
    unsigned int corresponding_block_right_border = corresponding_block_left_border + group_size;
    if (corresponding_block_left_border >= n) {
        corresponding_block_left_border = n;
    }
    if (corresponding_block_right_border >= n) {
        corresponding_block_right_border = n;
    }
    unsigned int l = corresponding_block_left_border - 1;
    unsigned int r = corresponding_block_right_border;
    const unsigned int input_value = input_data[i];

    while (r - l > 1) {
        const unsigned int m = (r + l) / 2;
        const unsigned int value = input_data[m];
        if (is_left) {
            if (value < input_value) {
               l = m;
            } else {
               r = m;
            }
        } else {
            if (value <= input_value) {
              l = m;
            } else {
              r = m;
          }
        }
    }
    const unsigned int output_block_position = (l + 1) - corresponding_block_left_border + (i - block_left_border);
    const unsigned int output_block_left_border = is_left ? block_left_border : corresponding_block_left_border;
    output_data[output_block_left_border + output_block_position] = input_value;
}
