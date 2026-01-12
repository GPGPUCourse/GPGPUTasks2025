#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
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
    const int idx = get_global_id(0);

    if (idx >= n) return;

    const int full_width = 2 * sorted_k;
    const int block_start = (idx / full_width) * full_width;

    const int split_idx = min(block_start + sorted_k, n);
    const int block_end = min(block_start + full_width, n);

    const bool is_left_part = (idx < split_idx);
    const uint my_val = input_data[idx];

    int search_base = is_left_part ? split_idx : block_start;
    int search_len  = is_left_part ? (block_end - split_idx) : (split_idx - block_start);

    while (search_len > 0) {
        int half_len = search_len >> 1;
        int mid = search_base + half_len;
        uint other_val = input_data[mid];

        bool move_right = (other_val < my_val) || (is_left_part && other_val == my_val);

        if (move_right) {
            search_base = mid + 1;
            search_len -= (half_len + 1);
        } else {
            search_len = half_len;
        }
    }

    int found_offset = search_base - (is_left_part ? split_idx : block_start);
    int local_offset = idx - (is_left_part ? block_start : split_idx);
    int final_pos = block_start + local_offset + found_offset;

    if (final_pos < n) {
        output_data[final_pos] = my_val;
    }
}
