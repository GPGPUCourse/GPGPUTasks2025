#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void merge_sort(
    __global const uint* restrict src,
    __global       uint* restrict dst,
                   int  half_width,
                   int  total_size)
{
    const int idx = get_global_id(0);

    if (idx >= total_size) return;

    const int full_width = 2 * half_width;
    const int block_start = (idx / full_width) * full_width;

    const int split_idx = min(block_start + half_width, total_size);
    const int block_end = min(block_start + full_width, total_size);

    const bool is_left_part = (idx < split_idx);
    const uint my_val = src[idx];

    int search_base = is_left_part ? split_idx : block_start;
    int search_len  = is_left_part ? (block_end - split_idx) : (split_idx - block_start);

    while (search_len > 0) {
        int half = search_len >> 1;
        int mid = search_base + half;
        uint other_val = src[mid];

        bool move_right = (other_val < my_val) || (is_left_part && other_val == my_val);

        if (move_right) {
            search_base = mid + 1;
            search_len -= (half + 1);
        } else {
            search_len = half;
        }
    }

    int found_offset = search_base - (is_left_part ? split_idx : block_start);
    int local_offset = idx - (is_left_part ? block_start : split_idx);
    int final_pos = block_start + local_offset + found_offset;

    if (final_pos < total_size) {
        dst[final_pos] = my_val;
    }
}
