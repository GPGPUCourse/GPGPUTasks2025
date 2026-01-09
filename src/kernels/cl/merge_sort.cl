#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

int get_merge_index(__global const uint* input_data, int bucket_size, int block_start_idx_inclusive, uint value, bool prefer_left_on_equal) {
    int left = block_start_idx_inclusive - 1;
    int right = block_start_idx_inclusive + bucket_size;

    while (left + 1 < right) {
        int m = left + ((right - left) >> 1);
        if (input_data[m] < value || (input_data[m] == value && prefer_left_on_equal)) {
            left = m;
        } else {
            right = m;
        }
    }
    return right - block_start_idx_inclusive;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* in,
    __global       uint* out,
                   int  pow,
                   int  n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    uint bucket_size = 1u << pow;

    uint pair_start = (i >> (pow + 1)) << (pow + 1);
    uint is_right = (i >> pow) & 1u;         

    uint block_start = (i >> pow) << pow;
    uint in_block = i - block_start;

    uint left_start  = pair_start;
    uint right_start = pair_start + bucket_size;

    uint left_len = min(bucket_size, n - left_start);
    uint right_len = (right_start < n) ? min(bucket_size, n - right_start) : 0;

    if (right_len == 0) { 
        out[i] = in[i]; 
        return; 
    }

    uint value = in[i];
    uint offset = in_block + get_merge_index(in, (is_right ? left_len  : right_len), (is_right ? left_start : right_start), value, !is_right);
    out[pair_start + offset] = value;
}
