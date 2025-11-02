#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

static inline int bin_search(__global const uint* data, int start, int len, uint val, bool strong_cmp)
{
    int l = 0;
    int r = len;

    while (l < r) {
        int m = (l + r) / 2;
        if ((strong_cmp && (data[start + m] < val))
            || (!strong_cmp && (data[start + m] <= val))) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    // DONE

    const int global_id = get_global_id(0);

    if (global_id >= n) {
        return;
    }

    const int block_size = 2 * sorted_k;
    const int block_id = global_id / block_size;
    const int block_start = block_id * block_size;

    const int left_start = block_start;
    const int left_size = max(0, min(sorted_k, n - left_start));
    const int right_start = left_start + sorted_k;
    const int right_size = max(0, min(sorted_k, n - right_start));

    const bool in_right = (global_id - block_start >= left_size);
    const uint value = input_data[global_id];

    int my_start = in_right ? right_start : left_start;
    int other_start = in_right ? left_start : right_start;
    int other_size = in_right ? left_size : right_size;

    int out_position = left_start
        + (global_id - my_start)
        + bin_search(input_data, other_start, other_size, value, !in_right);

    output_data[out_position] = value;
}
