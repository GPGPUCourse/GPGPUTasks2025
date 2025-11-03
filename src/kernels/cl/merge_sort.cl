#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n,
                   int  chunk_size)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int first_part_index = i - i % (chunk_size * 2);
    int second_part_index = first_part_index + chunk_size;
    int local_shift = i % chunk_size;
    if (second_part_index >= n) {
        output_data[i] = input_data[i];
        return;
    }
    int l = -1;
    int r = chunk_size;
    int addition;
    bool strict_less;
    if (i < second_part_index) {
        addition = second_part_index;
        strict_less = true;
    }
    else {
        addition = first_part_index;
        strict_less = false;
    }
    while (r - l > 1) {
        int m = (r + l) / 2;
        int index_m = m + addition;
        if (index_m < n && ((input_data[index_m] < input_data[i] && strict_less) || (input_data[index_m] <= input_data[i] && !strict_less))) {
            l = m;
        }
        else {
            r = m;
        }
    }
    output_data[first_part_index + local_shift + r] = input_data[i];
}
