#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const uint i = get_global_id(0);
    if (i > n) {
        return;
    }
    const uint element = input_data[i];
    const uint block_id = i / sorted_k;
    const uint block_offset = i % sorted_k;
    const uint block_start = (block_id & (~1)) * sorted_k;
    const bool block_is_left = block_id % 2 == 0;
    int l, r;
    if (block_is_left) {
        l = block_start + sorted_k;
        r = min(block_start + 2 * sorted_k, n);
        
    } else {
        l = block_start;
        r = min(block_start + sorted_k, n);
    }
    while (r > l) {
        int m = (r + l) / 2;
        if (input_data[m] < element || (!block_is_left && input_data[m] == element)) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    uint target_offset = i + l - 2 * block_start - sorted_k;
    // printf("element %d; target offset %d; i: %d \n", element, target_offset, i);
    output_data[block_start + target_offset] = element;   
}
