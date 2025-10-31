#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  len,
                   int  n)
{
    const unsigned int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    if (len >= n) {
        output_data[idx] = input_data[idx];
        return;
    }

    const unsigned int block_idx = idx / len;
    const unsigned find_block_idx = (block_idx % 2) ? block_idx - 1 : block_idx + 1;
    int left = find_block_idx * len - 1;


    if (find_block_idx * len >= n) {
        output_data[idx] = input_data[idx];
        return;
    }

    int right = (find_block_idx + 1) * len >= n ? n : (find_block_idx + 1) * len;

    const unsigned int to_find = input_data[idx];
    //printf("%d %d %d \n", to_find, left, right);

    while (left < right - 1) {
        unsigned int mid = (left + right) / 2;
        if (input_data[mid] < to_find + (block_idx % 2)) {
            left = mid;
        } else {
            right = mid;
        }
    }
    const unsigned int find_block_less = (right - find_block_idx * len);
    const unsigned int block_less = (idx - block_idx * len);
    const unsigned int start = block_idx % 2 ? find_block_idx * len : block_idx * len;
    const unsigned int to = start + find_block_less + block_less;
    //printf("%d %d %d %d\n", to_find, start, find_block_less, block_less);

    //printf("%d %d\n", to_find, to);
    output_data[to] = to_find;
}
