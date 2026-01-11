#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

inline uint count_less(__global const uint* arr, uint len, uint val) {
    uint l = 0, r = len;
    while (l < r) {
        uint m = (l + r) >> 1;
        if (arr[m] < val) l = m + 1; else r = m;
    }
    return l;
}

inline uint count_less_equal(__global const uint* arr, uint len, uint val) {
    uint l = 0, r = len;
    while (l < r) {
        uint m = (l + r) >> 1;
        if (arr[m] <= val) l = m + 1; else r = m;
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
    const unsigned int i = get_global_id(0);
    if (i >= (uint) n)
        return;

    const uint block_size  = (uint) (sorted_k << 1);
    const uint block_start = (i / block_size) * block_size;

    const uint left_begin  = block_start;
    const uint left_end    = min(block_start + (uint)sorted_k, (uint)n);
    const uint right_begin = left_end;

    if (right_begin >= (uint) n) {
        output_data[i] = input_data[i];
        return;
    }

    const uint right_end = min(block_start + block_size, (uint) n);

    const uint left_len   = left_end  - left_begin;
    const uint right_len  = right_end - right_begin;

    uint val = input_data[i];
    bool in_left = (i < right_begin);

    if (in_left) {
        uint li   = i - left_begin;
        uint rank = li + count_less(input_data + right_begin, right_len, val);
        uint out_idx = block_start + rank;
        output_data[out_idx] = val;
    } else {
        uint rj   = i - right_begin;
        uint rank = rj + count_less_equal(input_data + left_begin, left_len, val);
        uint out_idx = block_start + rank;
        output_data[out_idx] = val;
    }
}
