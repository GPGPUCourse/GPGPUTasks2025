#include "helpers/rassert.cl"
#include "../defines.h"

void copy_remaining(__global const uint* input_data,
                    __global       uint* output_data,
                               int  start,
                               int* index,
                               int  size,
                               int* pos)
{
    while (*index < size) {
        output_data[*pos] = input_data[start + *index];
        (*index)++;
        (*pos)++;
    }
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int block_pair_id = get_global_id(0);

    int left_start = block_pair_id * 2 * sorted_k;
    int right_start = left_start + sorted_k;
    int merge_end = min(left_start + 2 * sorted_k, n);

    if (left_start >= n) {
        return;
    }

    int left_size = min(sorted_k, n - left_start);
    int right_size = (right_start < n) ? min(sorted_k, n - right_start) : 0;

    int i = 0;
    int j = 0;
    int pos = left_start;

    while (i < left_size && j < right_size) {
        uint left_val = input_data[left_start + i];
        uint right_val = input_data[right_start + j];

        if (left_val <= right_val) {
            output_data[pos] = left_val;
            i++;
        } else {
            output_data[pos] = right_val;
            j++;
        }
        pos++;
    }

    copy_remaining(input_data, output_data, left_start, &i, left_size, &pos);
    copy_remaining(input_data, output_data, right_start, &j, right_size, &pos);
}
