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

    const uint key_v = input_data[i];
    size_t bucket_idx = i / (2 * sorted_k);
    size_t bucket_start = 2 * sorted_k * bucket_idx;

    size_t left_start = bucket_start;
    size_t left_end = (left_start + sorted_k < n) ? left_start + sorted_k : n;

    size_t right_start = left_end;
    size_t right_end = (right_start + sorted_k < n) ? right_start + sorted_k : n;
    if (right_start >= n) {
        output_data[i] = key_v;
        return;
    }

    size_t pos = UINT_MAX;
    if (i >= right_start) { // is in right
        size_t l = left_start;
        size_t r = left_end;
        while (l < r) {
            size_t m = (l + r) / 2;
            if (input_data[m] <= key_v) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        size_t i_left = l - left_start;
        size_t i_right = i - right_start;
        pos = left_start + i_left + i_right;
    } else {
        size_t l = right_start;
        size_t r = right_end;
        while (l < r) {
            size_t m = (l + r) / 2;
            if (input_data[m] < key_v) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        size_t i_left = i - left_start;
        size_t i_right = l - right_start;
        pos = left_start + i_left + i_right;       
    }
    
    output_data[pos] = key_v;
}
