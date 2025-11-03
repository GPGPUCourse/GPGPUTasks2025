#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_big(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
     __local uint buffer[GROUP_SIZE];

    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int block_num = index / sorted_k;
    const unsigned int num_in_block = index % sorted_k;
    const unsigned int k = (block_num >> 1) * (sorted_k << 1) + num_in_block;

    if (index >= n) {
        return;
    }

    unsigned int i = index - num_in_block;
    uint value = input_data[index];

    if (block_num & 1) {
        unsigned int l = i - sorted_k;
        unsigned int r = i;

        unsigned int m = l + num_in_block + (sorted_k >> 4);
        if (m >= r) {
            m = r - (sorted_k >> 2);
        }

        while (r - l > 1) {
            uint current = input_data[m];

            if (current > value) {
                r = m;
            } else {
                l = m;
            }

            m = (l + r) >> 1;
        }

        if (l == i - sorted_k) {
            uint current = input_data[l];
            if (current <= value) {
                output_data[k + l + sorted_k - i + 1] = value;
            } else {
                output_data[k + l + sorted_k - i] = value;
            }
        } else {
            output_data[k + l + sorted_k - i + 1] = value;
        }


    } else {
        unsigned int l = i + sorted_k;

        if (l >= n) {
            output_data[k] = value;
            return;
        }

        unsigned int r = l + sorted_k;

        if (r > n) {
            r = n;
        }

        unsigned int m = l + num_in_block + (sorted_k >> 4);
        if (m >= r) {
            m = r - (sorted_k >> 2);
        }

        while (r - l > 1) {
            unsigned int m = (l + r) >> 1;
            uint current = input_data[m];
            if (current >= value) {
                r = m;
            } else {
                l = m;
            }

            m = (l + r) >> 1;
        }

        if (l == i + sorted_k) {
            uint current = input_data[l];
            if (current < value) {
                output_data[k + l - i - sorted_k + 1] = value;
            } else {
                output_data[k + l - i - sorted_k] = value;
            }
        } else {
            output_data[k + l - i - sorted_k + 1] = value;
        }
    }
}
