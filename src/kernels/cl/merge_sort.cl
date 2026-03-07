#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
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
    const unsigned int i = get_global_id(0);

    if (i >= n)
        return;

    const uint block_idx = i / sorted_k;
    const uint is_left = (block_idx % 2 == 0);
    const uint local_idx = i % sorted_k;

    if (is_left) {
        const uint right_block = block_idx + 1;
        const uint right_start = right_block * sorted_k;

        if (right_start >= n) {
            output_data[i] = input_data[i];
            return;
        }

        uint left = right_start;
        uint right = min(right_start + sorted_k, (uint)n);

        while (left < right) {
            uint mid = (left + right) / 2;

            if (input_data[mid] < input_data[i]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        const uint out_idx = block_idx * sorted_k + local_idx + (left - right_start);

        if (out_idx < n) {
            output_data[out_idx] = input_data[i];
        }

        return;
    }

    const uint left_block = block_idx - 1;
    const uint left_start = left_block * sorted_k;

    if (block_idx == 0) {
        output_data[i] = input_data[i];
        return;
    }

    uint left = left_start;
    uint right = min(left_start + sorted_k, (uint)n);

    while (left < right) {
        uint mid = (left + right) / 2;

        if (input_data[mid] <= input_data[i]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    const uint out_idx = left_block * sorted_k + local_idx + (left - left_start);

    if (out_idx < n) {
        output_data[out_idx] = input_data[i];
    }
}