#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

int binary_search(
    __global const uint* data,
                   int n,
                   uint target,
                   bool inclusive)
{
    int left = 0;
    int right = n - 1;
    while (left <= right) {
        int middle = (left + right) / 2;
        if (data[middle] < target || (data[middle] == target && inclusive)) {
            left = middle + 1;
        } else {
            right = middle - 1;
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
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int my_sort_group_idx = i / sorted_k;
    const unsigned int neigh_sort_group_idx = my_sort_group_idx + ((my_sort_group_idx & 1) == 0 ? 1 : -1);

    const unsigned int my_num_idx = i - my_sort_group_idx * sorted_k;
    const unsigned int neigh_num_idx = binary_search(
            input_data + neigh_sort_group_idx * sorted_k,
            min(sorted_k, (int)(n - neigh_sort_group_idx * sorted_k)),
            input_data[i],
            (my_sort_group_idx & 1) == 1
    );

    const unsigned int result_idx = min(my_sort_group_idx, neigh_sort_group_idx) * sorted_k + my_num_idx + neigh_num_idx;
    if (result_idx < n) {
        output_data[result_idx] = input_data[i];
    }
}
