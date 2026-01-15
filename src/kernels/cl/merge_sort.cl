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
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    int pair = i / (2 * sorted_k);
    int left_start = pair * 2 * sorted_k;
    int right_start = left_start + sorted_k;
    int left_end = min(right_start, n);
    int right_end = min(left_start + 2 * sorted_k, n);

    if (i < right_start && i < left_end) {
        unsigned int x = input_data[i];
        int l = right_start, r = right_end;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (input_data[m] < x) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        int count = l - right_start;
        int pos_in_left = i - left_start;
        int out_idx = left_start + pos_in_left + count;
        output_data[out_idx] = x;
    } else if (i >= right_start && i < right_end) {
        unsigned int x = input_data[i];
        int l = left_start, r = left_end;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (input_data[m] <= x) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        int count = l - left_start;
        int pos_in_right = i - right_start;
        int out_idx = left_start + count + pos_in_right;
        output_data[out_idx] = x;
    }
}
