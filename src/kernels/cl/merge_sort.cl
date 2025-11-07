#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  bit,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    int out_start = (i >> (bit + 1)) << (bit + 1);
    int mid       = out_start + (1 << bit);
    int out_end   = out_start + (2 << bit);

    if (mid < n) {
        int l, r;
        if (i < mid) {
            l = mid;
            r = out_end;
            if (r > n) r = n;
        } else {
            l = out_start;
            r = mid;
        }

        --l; // l points to <x, r points to >x
        while (r - l > 1) {
            int m = (l + r) >> 1;
            if (input_data[m] < input_data[i] || (i >= mid && input_data[m] == input_data[i])) {
                l = m;
            } else {
                r = m;
            }
        }
        int where = out_start;
        if (i < mid) {
            where += (i - out_start) + (r - mid);
        } else {
            where += (i - mid) + (r - out_start);
        }
        output_data[where] = input_data[i];
    } else {
        output_data[i] = input_data[i];
    }
}
