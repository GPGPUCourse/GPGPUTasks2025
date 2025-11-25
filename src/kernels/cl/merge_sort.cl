#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

static inline int lower_bound(__global const uint* arr, int base, int len, uint value) {
    int l = 0, h = len;
    while (l < h) {
        int mid = (l + h) / 2;
        if (arr[base + mid] < value)
            l = mid + 1;
        else
            h = mid;
    }
    return l;
}

static inline int upper_bound(__global const uint* arr, int base, int len, uint value) {
    int l = 0, h = len;
    while (l < h) {
        int mid = (l + h) / 2;
        if (arr[base + mid] <= value)
            l = mid + 1;
        else
            h = mid;
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
    if (i >= n) return;

    const int block_size = 2 * sorted_k;
    const int block_id = i / block_size;

    const int left_start  = block_id * block_size;
    const int right_start = left_start + sorted_k;

    const int left_len  = (left_start < n) ? min(sorted_k, n - left_start) : 0;
    const int right_len = (right_start < n) ? min(sorted_k, n - right_start) : 0;

    const int pos = i - left_start;
    const uint value = input_data[i];

    if (pos < left_len) {
        int count = lower_bound(input_data, right_start, right_len, value);
        output_data[left_start + pos + count] = value;
    } else {
        int count = upper_bound(input_data, left_start, left_len, value);
        int pos_right = pos - left_len;
        output_data[left_start + pos_right + count] = value;
    }
}
