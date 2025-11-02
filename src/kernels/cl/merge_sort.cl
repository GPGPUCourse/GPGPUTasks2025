#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // только для IDE
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
    if (gid >= n) 
        return;

    const uint len_pair  = sorted_k << 1;
    const uint base_pair = gid & ~(len_pair - 1);
    const uint left_begin  = base_pair;
    const uint left_end  = min(base_pair + sorted_k, n);

    const uint right_begin = left_end;
    const uint right_end = min(base_pair + len_pair, n);

    if (right_begin >= right_end) {
        output_data[gid] = input_data[gid];
        return;
    }

    const uint value  = input_data[gid];
    const bool inLeft = (gid < right_begin);

    uint lo, hi, mid, rank;

    if (inLeft) {
        lo = right_begin;
        hi = right_end;
        while (lo < hi) {
            mid = lo + ((hi - lo) >> 1);
            const uint x = input_data[mid];
            if (x < value)
                lo = mid + 1; 
            else 
                hi = mid;
        }
        rank = lo - right_begin;
        const uint offset = gid - left_begin;
        const uint pos = base_pair + offset + rank;
        output_data[pos] = value;
    } else {
        lo = left_begin;
        hi = left_end;
        while (lo < hi) {
            mid = lo + ((hi - lo) >> 1);
            const uint x = input_data[mid];
            if (x <= value)
                lo = mid + 1;
            else 
                hi = mid;
        }
        rank = lo - left_begin;
        const uint offset = gid - right_begin;
        const uint pos = base_pair + offset + rank;
        output_data[pos] = value;
    }
}
