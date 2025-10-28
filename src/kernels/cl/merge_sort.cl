#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
              int  sorted_k,
                   int  n)
{
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int block_size = (sorted_k * 2);
    const int block_id = i / block_size;

    const int block_start = block_id * block_size;


    const int block_right_start = block_start + sorted_k;

    if (block_right_start >= n) {
        output_data[i] = input_data[i];
        return;
    }


    const bool is_right = i >= block_right_start;

    int l = (is_right ? block_start : block_right_start) - 1;

    int r = (is_right ? block_start + sorted_k : min(block_right_start + sorted_k, n));

    uint val = input_data[i];


    while (r - l > 1) {
        const int m = (l + r) / 2;
        if (is_right && input_data[m] > val || !is_right && input_data[m] >= val) {
            r = m;
        } else {
            l = m;
        }
    }

    int index = block_start + (is_right ? (i - block_right_start) + (r - block_start) : (i - block_start) + (r - block_right_start));

    output_data[index] = val;
}
