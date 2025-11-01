#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,  // [start; min(start + l_k, n) ) 
    __global       uint* output_data, // [start; min(start + l_k, n) )
                   int  n,
                   int  step_k,
                   int  start)
{
    const unsigned int i = get_global_id(0);
    const unsigned int local_i = get_local_id(0);
    const int bucket = local_i / step_k;
    const unsigned left_bound = start + step_k * bucket;
    const unsigned right_bound = min(n, start + step_k * (bucket + 1));
    const unsigned middle = (left_bound + start + step_k * (bucket + 1)) >> 1;
    unsigned int is_left = 0;
    unsigned int L, R, shift, loc_shift;
    if (i >= right_bound || i < left_bound) return;
    if (i < middle) {
        // left
        L = middle;
        R = right_bound;
    } else {
        L = left_bound;
        R = middle;
        is_left = 1;
    }
    shift = i - L;
    const unsigned int item = input_data[i] + is_left;
    while (R > L) {
        unsigned int M = (R + L) >> 1;
        if (item <= input_data[M]) {
            R = M;
        } else {
            L = M + 1;
        }
    }
    output_data[i - middle + L] = input_data[i];
}
