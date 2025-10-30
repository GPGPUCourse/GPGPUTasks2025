#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // только для IDE
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint gid = get_global_id(0);
    const uint pair_span = 2 * sorted_k;
    const uint base = gid * pair_span;

    if (base >= n) 
        return;

    const uint left_begin = base;
    const uint left_end = min(base + sorted_k, (uint)n);
    const uint right_begin = left_end;
    const uint right_end = min(base + pair_span, (uint)n);

    if (right_begin >= right_end) {
        for (uint i = left_begin; i < left_end; ++i)
            output_data[i] = input_data[i];
        return;
    }

    uint i = left_begin;
    uint j = right_begin;
    uint k = base;

    while (i < left_end && j < right_end) {
        uint a = input_data[i];
        uint b = input_data[j];
        if (a <= b) {
            output_data[k++] = a;
            ++i;
        } else {
            output_data[k++] = b;
            ++j;
        }
    }
    while (i < left_end)
        output_data[k++] = input_data[i++];
    while (j < right_end)
        output_data[k++] = input_data[j++];
}
