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

    if (i >= n) {
        return;
    }

    uint el = input_data[i];
    uint block_id = i / sorted_k;
    uint block_start = sorted_k * (block_id & (~1));
    bool block_is_left = block_id % 2 == 0;
    bool block_is_right = !block_is_left;


    int l = block_start;
    int r = min(n, block_start + sorted_k);
    if (block_is_left) {
        l += sorted_k;
        r = min(n, block_start + 2 * sorted_k);
    }
    while (r > l) {
        int m = (r + l) >> 1;
        if (block_is_right && input_data[m] == el || (input_data[m] < el)) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    uint target_offset = i + l - 2 * block_start - sorted_k;
    output_data[block_start + target_offset] = el;
}