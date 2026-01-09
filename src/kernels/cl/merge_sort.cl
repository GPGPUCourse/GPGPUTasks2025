#include "helpers/rassert.cl"
#include "../defines.h"

int binary_search_lower(__global const uint* arr, int start, int size, uint value)
{
    int left = 0;
    int right = size;

    while (left < right) {
        int mid = (left + right) / 2;
        if (arr[start + mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

int binary_search_upper(__global const uint* arr, int start, int size, uint value)
{
    int left = 0;
    int right = size;

    while (left < right) {
        int mid = (left + right) / 2;
        if (arr[start + mid] <= value) {
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
    const int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    int block_id = gid / (2 * sorted_k);
    int left_start = block_id * 2 * sorted_k;
    int right_start = left_start + sorted_k;
    int left_end = min(right_start, n);
    int right_end = min(right_start + sorted_k, n);

    uint value = input_data[gid];
    int output_pos;

    if (gid >= left_start && gid < left_end) {
        int rank_in_left = gid - left_start;

        int rank_in_right = binary_search_lower(
            input_data, right_start, right_end - right_start, value
        );

        output_pos = left_start + rank_in_left + rank_in_right;
    }
    else if (gid >= right_start && gid < right_end) {
        int rank_in_right = gid - right_start;

        int rank_in_left = binary_search_upper(
            input_data, left_start, left_end - left_start, value
        );

        output_pos = left_start + rank_in_right + rank_in_left;
    }
    else {
        output_pos = gid;
    }

    output_data[output_pos] = value;
}
