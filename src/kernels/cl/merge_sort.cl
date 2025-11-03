#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int bucket_index = i / sorted_k;
    const int bucket_start_index = bucket_index * sorted_k;
    const int neighbour_bucket_start_index = (bucket_index ^ 1) * sorted_k;
    const bool is_left_bucket = bucket_start_index < neighbour_bucket_start_index;
    if (neighbour_bucket_start_index >= n) {
        output_data[i] = input_data[i];
        return;
    }
    // non inclusive
    const int right_search_border = min(sorted_k, n - neighbour_bucket_start_index) + 1;
    const int x = input_data[i];

    printf("Right search border for index %d is %d\n", i, right_search_border);
    int left = 0, right = right_search_border, mid, y;
    while (right - left > 1) {
        mid = (left + right) / 2;
        y = input_data[neighbour_bucket_start_index + mid - 1];
        if (y > x || (is_left_bucket && y == x)) {
            right = mid;
        } else {
            left = mid;
        }
    }
    const int count_less_in_neighbour_bucket = left;

    const int merged_start_index = min(bucket_start_index, neighbour_bucket_start_index);
    const int start_index = merged_start_index + i - bucket_start_index;
    const int new_index = start_index + count_less_in_neighbour_bucket;
    printf("Placed input_data[%d] = %d by index %d (neighbour index = %d, start index = %d)\n",
        i, x, new_index, count_less_in_neighbour_bucket, start_index);
    output_data[new_index] = x;
}
