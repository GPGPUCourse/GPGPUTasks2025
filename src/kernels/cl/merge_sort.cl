#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint upper_bound(__global const uint* data, uint base, uint len, uint val) {
    uint left = 0;
    uint right = len;
    while (left < right) {
        uint mid = left + (right - left) / 2;
        if (data[base + mid] <= val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

uint lower_bound(__global const uint* data, uint base, uint len, uint val) {
    uint left = 0;
    uint right = len;
    while (left < right) {
        uint mid = left + (right - left) / 2;
        if (data[base + mid] < val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint sorted_k,
                   uint n)
{
    const uint idx = get_global_id(0);
    
    if (idx >= n) {
        return;
    }

    uint pair_size = 2 * sorted_k;
    uint pair_base = (idx / pair_size) * pair_size;
    
    uint left_start = pair_base;
    uint left_end = min(pair_base + sorted_k, n);
    uint left_len = left_end - left_start;
    
    uint right_start = left_end;
    uint right_end = min(pair_base + pair_size, n);
    uint right_len = right_end - right_start;

    if (right_len == 0) {
        output_data[idx] = input_data[idx];
        return;
    }

    uint value = input_data[idx];
    
    uint count_other = 0;
    uint local_index = 0;

    if (idx < right_start) {
        local_index = idx - left_start;
        
        count_other = lower_bound(input_data, right_start, right_len, value);
        
        const uint pos = pair_base + local_index + count_other;
        if (pos < n)
            output_data[pos] = value;

    } else {
        local_index = idx - right_start;
        
        count_other = upper_bound(input_data, left_start, left_len, value);
        
        const uint pos = pair_base + local_index + count_other;
        if (pos < n)
            output_data[pos] = value;

    }
}