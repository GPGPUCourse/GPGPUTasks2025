#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }

    const uint group_index = index / sorted_k;
    uint neighbor_group_index = group_index + 1;
    if ((group_index & 1) != 0) {
        neighbor_group_index = group_index - 1;
    }

    const uint local_pos = index - group_index * sorted_k;
    const uint neighbor_group_offset = neighbor_group_index * sorted_k;
    

    if (neighbor_group_index * sorted_k >= (uint)n) {
        output_data[index] = input_data[index];
        return;
    }

    const uint target_value = input_data[index];
    int l = 0;
    int r = (int)min(sorted_k, (int)(n - neighbor_group_offset)) - 1;

    while (l <= r) {
        int m = l + ((r - l) / 2);
        uint val = input_data[neighbor_group_offset + m];
        if (val < target_value || ((group_index & 1) != 0 && val == target_value)) {
            l = m + 1;
        }
        else {
            r = m - 1;
        }
    }

    const uint merge_start = min(group_index, neighbor_group_index) * sorted_k;
    const uint target_pos = merge_start + local_pos + (uint)l;
    if (target_pos < n) {
        output_data[target_pos] = target_value;
    }
}
