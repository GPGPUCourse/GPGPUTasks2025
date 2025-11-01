#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline int lower_bound_global(__global const uint* arr, int lo, int hi, uint x) {
    int l = lo, r = hi;
    while (l < r) {
        int m = l + ((r - l) >> 1);
        uint v = arr[m];
        if (v < x) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

static inline int upper_bound_global(__global const uint* arr, int lo, int hi, uint x) {
    int l = lo, r = hi;
    while (l < r) {
        int m = l + ((r - l) >> 1);
        uint v = arr[m];
        if (v <= x) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int glob_id = get_global_id(0);
    if (glob_id >= n)
        return;

    const unsigned int me = input_data[glob_id];

    unsigned int two_block_idx = glob_id / (sorted_k * 2);
    unsigned int block_idx = glob_id / sorted_k;

    unsigned int two_block_start = two_block_idx * (sorted_k * 2);
    unsigned int block_start = block_idx * sorted_k;

    const unsigned int l0 = two_block_start;
    const unsigned int r0 = MIN(l0 + sorted_k, n);
    const unsigned int l1 = r0;
    const unsigned int r1 = MIN(l1 + sorted_k, n);
    
    if (two_block_start == block_start) {
        unsigned int lower_bound = lower_bound_global(input_data, l1, r1, me);
        unsigned int new_idx = l0 + (glob_id - l0) + (lower_bound - l1);
        output_data[new_idx] = me;
    }
    else {
        unsigned int upper_bound = upper_bound_global(input_data, l0, r0, me);
        unsigned int new_idx = l0 + (glob_id - l1) + (upper_bound - l0);
        output_data[new_idx] = me;
    }

}
