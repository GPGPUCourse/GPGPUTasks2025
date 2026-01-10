#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


inline int binary_search(__global const uint* arr, int left, int right, uint value, int is_left_block)
{
    while (left < right) {
        int mid = (left + right) / 2;
        int go_right = is_left_block ? (arr[mid] < value) : (arr[mid] <= value);
        if (go_right) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

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

    const int block_size = 2 * sorted_k;
    
    const int block_idx = i / block_size;
    
    const int block_start = block_idx * block_size;
    
    const int a_start = block_start;
    const int a_end = min(block_start + sorted_k, n);

    const int b_start = block_start + sorted_k;
    const int b_end = min(block_start + block_size, n);

    if (b_start >= n) {
        output_data[i] = input_data[i];
        return;
    }

    const int is_left_block = (i < b_start);
    
    const uint value = input_data[i];
    
    int result_pos;
    
    if (is_left_block) {
        const int pos_in_a = i - a_start;
        const int count_in_b = binary_search(input_data, b_start, b_end, value, 1) - b_start;
        result_pos = block_start + pos_in_a + count_in_b;
    } else {
        const int pos_in_b = i - b_start;
        const int count_in_a = binary_search(input_data, a_start, a_end, value, 0) - a_start;
        result_pos = block_start + pos_in_b + count_in_a;
    }
    
    output_data[result_pos] = value;
}
