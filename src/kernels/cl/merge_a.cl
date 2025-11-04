#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_a(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint n, 
                   uint block_size)
{
    const unsigned int i = get_global_id(0) / block_size * 2 * block_size + get_global_id(0) % block_size;
    
    if (i >= n) {
        return;
    }

    const uint a_index = i % block_size;
    const uint a_start = i - a_index;
    const uint b_start = a_start + block_size;
    const uint b_end = min(b_start + block_size, n);

    if(b_start >= n) {
        output_data[a_start + a_index] = input_data[a_start + a_index];
        return;
    }

    int l = 0, r = b_end - b_start;
    while(l + 1 < r) {
        int m = (l + r) / 2;
        if (input_data[b_start + m] >= input_data[a_start + a_index]) {
            r = m;
        } else {
            l = m;
        }
    }

    if (input_data[a_start + a_index] <= input_data[b_start]) {
        r = 0;
    }

    output_data[a_start + a_index + r] = input_data[a_start + a_index];
}
