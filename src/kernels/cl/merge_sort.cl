#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

int binary_search(__global const uint* input, int block_size, int offset, uint value, bool is_left) {
    int left = -1;
    int right = block_size;
    while (left + 1 < right) { 
        int mid = (left + right) / 2;
        if (input[offset + mid] < value || (input[offset + mid] == value && is_left)) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return right;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k, // pow
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    int block_size = 1 << sorted_k;
    int block_id = i / block_size;
    bool is_cur_block_left = 1 - (block_id & 1);
    
    int block_start = block_id * block_size;
    int pair_block_start = block_start + (is_cur_block_left ? block_size : -block_size);
    int pair_count = max(0, min(block_size, n - pair_block_start));

    int bs = binary_search(input_data, pair_count, pair_block_start, input_data[i], is_cur_block_left);

    output_data[i + bs + (is_cur_block_left ? 0 : -block_size)] = input_data[i];
}
