#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_small(
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

        output_data[k + i - begin] = value;
    } else {
        unsigned int diff = sorted_k - num_in_block;
        unsigned int index_diff = index + diff;
        unsigned int begin = local_index + diff;
        unsigned int i = begin;
        unsigned int end = begin + sorted_k;

        if (index_diff >= n) {
            end = begin;
        } else if (index_diff + sorted_k > n) {
            end = begin + (n - index_diff);
        }

        for (; i < end && buffer[i] < value; ++i) {
        }

        output_data[k + i - begin] = value;
    }
}
