#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


int bin_search(__global const uint* arr, int start, int end, int elem, bool left_side) {
    // если left_size == 1, то применяем правосторонний поиск
    while (start < end - 1) {
        int m = (start + end) / 2;
        bool condition;
        if (left_side) {
            condition = arr[m] < elem;
        } else {
            condition = arr[m] <= elem;
        }
        if (condition) {
            start = m;
        } else {
            end = m;
        }
    }
    if (left_side) {
        return start + 1;
    } else {
        return end;
    }
}

__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const unsigned int block_index = i / sorted_k;
    const unsigned int local_index = i % sorted_k;
    int merge_block_start_index, merge_block_end_index;

    bool left = (block_index % 2 == 0);

    if (left) {
        merge_block_start_index = (block_index + 1) * sorted_k - 1;
        merge_block_end_index = (block_index + 2) * sorted_k + 1;
    } else {
        merge_block_start_index = (block_index - 1) * sorted_k - 1;
        merge_block_end_index = block_index * sorted_k + 1;
    }
    if (merge_block_start_index >= n) {
        output_data[i] = input_data[i];
        return;
    }
    merge_block_end_index = min(merge_block_end_index, n + 1);


    int global_offset;
    if (left) {
        global_offset = block_index * sorted_k;
    } else {
        global_offset = (block_index - 1) * sorted_k;
    }

    if (sorted_k == 1) {
        if (left) {
            if (input_data[i] <= input_data[i + 1]) {
                output_data[i] = input_data[i];
            } else {
                output_data[i + 1] = input_data[i];
            }
        } else {
            if (input_data[i - 1] <= input_data[i]) {
                output_data[i] = input_data[i];
            } else {
                output_data[i - 1] = input_data[i];
            }
        }
        return;
    } else {
        int position_merge = bin_search(input_data, max(-1, merge_block_start_index), min(merge_block_end_index, n+1), input_data[i], left);
        if (position_merge == merge_block_end_index) {
            position_merge -= 1;
        }
        int output_index = global_offset + local_index + position_merge - merge_block_start_index - 1;
        if (output_index >= 0 && output_index < n) {
            output_data[output_index] = input_data[i];
        }
    }
}
