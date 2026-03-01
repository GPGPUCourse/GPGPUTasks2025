#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(__global const uint* input_data,
                         __global       uint* output_data,
                                        int   sorted_k,
                                        int   n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;
    int block_id = gid / sorted_k;
    int is_right = block_id % 2;
    int left_start = (block_id - is_right) * sorted_k;
    int right_start = min(left_start + sorted_k, n);
    int right_end = min(right_start + sorted_k, n);

    uint my_val = input_data[gid];

    if (is_right == 0) {
        int l = right_start;
        int r = right_end;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (input_data[m] < my_val) l = m + 1;
            else r = m;
        }
        int count = l - right_start;
        output_data[gid + count] = my_val;
    } else {
        int l = left_start;
        int r = right_start;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (input_data[m] <= my_val) l = m + 1;
            else r = m;
        }
        int count = l - left_start;
        output_data[left_start + count + gid - right_start] = my_val;
    }
}