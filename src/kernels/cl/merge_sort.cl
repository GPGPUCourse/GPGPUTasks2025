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
                   uint  n)
{
    const unsigned int index = get_global_id(0);
    if (index >= n) {
        return;
    }

    uint size = 1 << (sorted_k - 1);
    uint left_base = (index >> sorted_k) << sorted_k;
    uint border = left_base + size;
    uint right_base = border + size;

    if (border >= n) {
        output_data[index] = input_data[index];
        return;
    }
    uint l, r, begin, offset;
    if (index < border) {
        offset = left_base;
        begin = border - 1;
        r = min(right_base, n);
    } else {
        offset = border;
        begin = left_base - 1;
        r = border;
    }
    uint curr = input_data[index] + (index >= border);
    l = begin;

    while (r - l > 1) {
        uint mid = (l + r) / 2;
        if (input_data[mid] < curr) {
            l = mid;
        } else {
            r = mid;
        }
    }

    output_data[left_base + index - offset + l - begin] = input_data[index];
}
