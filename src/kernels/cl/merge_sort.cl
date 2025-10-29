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
     __local uint buffer[GROUP_SIZE];

    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    const unsigned int block_num = index / sorted_k;
    const unsigned int num_in_block = index % sorted_k;
    const unsigned int k = (block_num >> 1) * (sorted_k << 1);

    if (sorted_k <= (GROUP_SIZE >> 1)) {

        if (index < n) {
            buffer[local_index] = input_data[index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (index >= n) {
            return;
        }

        uint value = buffer[local_index];

        if (block_num & 1) {
            unsigned int end = local_index - num_in_block;
            unsigned int begin = end - sorted_k;
            unsigned int i = begin;

            for (; i < end && buffer[i] <= value; ++i) {
            }
            output_data[k + num_in_block + i - begin] = value;
        } else {
            unsigned int begin = local_index - num_in_block + sorted_k;
            unsigned int i = begin;
            unsigned int end = begin + sorted_k;

            if (index - num_in_block + sorted_k >= n) {
                output_data[k + num_in_block] = value;
                return;
            } else if (index - num_in_block + 2 * sorted_k > n) {
                end = begin + (n - (index - num_in_block + sorted_k));
            }

            for (; i < end && buffer[i] < value; ++i) {
            }
            output_data[k + num_in_block + i - begin] = value;
        }
    } else if (index < n) {
        unsigned int i = index - num_in_block;
        uint value = input_data[index];

        if (block_num & 1) {
            unsigned int l = i - sorted_k;
            unsigned int r = i;

            while (r - l > 1) {
                unsigned int m = (l + r) >> 1;
                uint current = input_data[m];
                if (current > value) {
                    r = m;
                } else {
                    l = m;
                }
            }

            uint current = input_data[l];
            if (current <= value) {
                ++l;
            }

            output_data[k + num_in_block + l + sorted_k - i] = value;
        } else {
            unsigned int l = i + sorted_k;

            if (l >= n) {
                output_data[k + num_in_block] = value;
                return;
            }

            unsigned int r = l + sorted_k;

            if (r > n) {
                r = n;
            }


            while (r - l > 1) {
                unsigned int m = (l + r) >> 1;
                uint current = input_data[m];
                if (current >= value) {
                    r = m;
                } else {
                    l = m;
                }
            }

            uint current = input_data[l];
            if (current < value) {
                ++l;
            }

            output_data[k + num_in_block + l - i - sorted_k] = value;
        }
    }
}
