#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    unsigned int x = 0;
    if (i < n) x = input_data[i];
    const int segment_size = 1 << sorted_k;
    const int segment_id = i / segment_size;
    const int segment_parity = segment_id % 2;

    int this_segment_start = segment_id * segment_size;
    int other_segment_start = segment_parity ? (segment_id - 1) * segment_size : (segment_id + 1) * segment_size;

    if (other_segment_start >= n) {
        if (i < n) output_data[i] = x;
        return;
    }

    int l = min(other_segment_start - 1, n);
    int r = min(other_segment_start + segment_size, n);

    while (r - l > 1) {
        unsigned int m = l + (r - l) / 2;
        unsigned int y = input_data[m];

        if ((segment_parity && y <= x) || (!segment_parity && y < x)) {
            l = m;
        } else {
            r = m;
        }
    }

    unsigned int left_segment_start = min(this_segment_start, other_segment_start);
    unsigned int j = left_segment_start + (r - other_segment_start) + (i - this_segment_start);

    if (i < n && j < n) output_data[j] = x;
}
