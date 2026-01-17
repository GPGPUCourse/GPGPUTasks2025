#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint bin_search(
    __global const uint* data,
    int l,
    int r,
    uint val,
    bool left)
{
    while (l < r - 1) {
        int mid = (l + r) / 2;
        uint cur = data[mid];
        if ((left && cur < val) || (!left && cur <= val)) {
            l = mid;
        } else {
            r = mid;
        }
    }
    uint res = r;
    if (left) {
        res = l + 1;
    }
    return res;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    bool left = (((idx / sorted_k) % 2) == 0);

    uint val = input_data[idx];

    uint start_idx_left, start_idx_right;
    if (left) {
        start_idx_left = (idx / sorted_k) * sorted_k;
        start_idx_right = ((idx / sorted_k) + 1) * sorted_k;
    } else {
        start_idx_right = (idx / sorted_k) * sorted_k;
        start_idx_left = ((idx / sorted_k) - 1) * sorted_k;
    }

    if (start_idx_right >= n) {
        output_data[idx] = val;
        return;
    }

    int len = sorted_k;

    if (left && start_idx_right + sorted_k >= n) {
        len = n - start_idx_right;
    }

    int l = start_idx_left - 1, r = start_idx_left + len;
    if (left) {
        l = start_idx_right - 1;
        r = start_idx_right + len;
    }

    uint output_idx = bin_search(input_data, l, r, val, left);

    uint result_idx = start_idx_left + idx - start_idx_right + output_idx - start_idx_left;
    if (left) {
        result_idx = start_idx_left + idx - start_idx_left + output_idx - start_idx_right;
    }

    output_data[result_idx] = val;
}
