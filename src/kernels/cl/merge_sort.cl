#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
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
    const uint gid = get_global_id(0);
    if ((int)gid >= n) return;

    const uint block = gid / sorted_k;
    const uint pos_in_seg = gid % sorted_k;

    const uint pair_index = block >> 1;
    const uint pair_base = pair_index * (sorted_k << 1);

    const uint left_start  = pair_base;
    const uint left_end    = min(pair_base + sorted_k, n);
    const uint right_start = pair_base + sorted_k;
    const uint right_end   = min(pair_base + (sorted_k << 1), n);

    const bool is_right = (block & 1u) != 0u;

    uint src_index;
    if (!is_right) {
        src_index = left_start + pos_in_seg;
        if (src_index >= left_end) {
            return;
        }
    } else {
        src_index = right_start + pos_in_seg;
        if (src_index >= right_end) {
            return;
        }
    }

    const uint val = input_data[src_index];

    uint lo = 0, hi = 0;
    uint count_other = 0;

    if (!is_right) {
        lo = right_start;
        hi = right_end;
        if (lo >= hi) {
            count_other = 0;
        } else {
            uint L = lo;
            uint R = hi;
            while (R - L > 1) {
                uint mid = (L + R) >> 1;
                uint cur = input_data[mid];
                if (cur < val) {
                    L = mid;
                } else {
                    R = mid;
                }
            }
            if (input_data[L] >= val) {
                count_other = L - lo;
            } else {
                if (R <= lo) {
                    count_other = 0;
                } else if (R >= hi) {
                    count_other = hi - lo;
                } else {
                    count_other = R - lo;
                }
            }
        }
        uint dest = pair_base + pos_in_seg + count_other;
        if (dest < n) output_data[dest] = val;
    } else {
        lo = left_start;
        hi = left_end;
        if (lo >= hi) {
            count_other = 0;
        } else {
            uint L = lo;
            uint R = hi;
            while (R - L > 1) {
                uint mid = (L + R) >> 1;
                uint cur = input_data[mid];
                if (cur <= val) {
                    L = mid;
                } else {
                    R = mid;
                }
            }
            if (input_data[L] > val) {
                count_other = (L > lo) ? (L - lo) : 0;
            } else {
                if (R <= lo) {
                    count_other = 0;
                } else if (R >= hi) {
                    count_other = hi - lo;
                } else {
                    count_other = R - lo;
                }
            }
        }
        uint dest = pair_base + count_other + pos_in_seg;
        if (dest < n) output_data[dest] = val;
    }
}
