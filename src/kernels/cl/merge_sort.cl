#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    unsigned int value = input_data[i];
    unsigned int block_index = i / sorted_k;
    bool is_left = (block_index % 2 == 0);

    unsigned int l = (block_index - !is_left) * sorted_k;
    unsigned int r = min(l + sorted_k, n);
    unsigned int m = min(r + sorted_k, n);


    // [l...][r...]m
    unsigned int bs_l = (is_left ? r : l) - 1;
    unsigned int bs_r = is_left ? m : r;
    unsigned int init_l = bs_l;

    while (bs_r - bs_l > 1) {
        unsigned int mid = (bs_l + bs_r) / 2;
        if (value > input_data[mid] || (!is_left && value == input_data[mid])) {
            bs_l = mid;
        } else {
            bs_r = mid;
        }
    }

    unsigned int shift = bs_l - init_l;
    unsigned int idx = is_left ? (i + shift) : (l + i - r + shift);
    
    // if (i == ?) {
    //     printf("i=%u, idx=%u shift=%u, l=%u, r=%u, m=%u, is_left=%u\n", i, idx, shift, l, r, m, is_left);
    // }
    output_data[idx] = value;
}
