#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

uint lower_bound(
    __global const uint* data,
    int size,
    uint key)
{
    int l = -1;
    int r = size;

    while (l < r - 1) {
        int mid = (l + r) / 2;

        if (data[mid] >= key) {
            r = mid;
        } else {
            l = mid;
        }
    }

    return r;
}

uint upper_bound(
    __global const uint* data,
    int size,
    uint key)
{
    int l = -1;
    int r = size;

    while (l < r - 1) {
        int mid = (l + r) / 2;

        if (data[mid] > key) {
            r = mid;
        } else {
            l = mid;
        }
    }

    return r;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    uint value = input_data[i];
    uint sub_block = i >> (sorted_k - 1);
    uint block = i >> sorted_k;

    uint block_start = block << sorted_k;
    uint index_in_sub_block = i - (sub_block << (sorted_k - 1));

    uint offset = 0;

    __global const uint* neighbor;
    int neighbor_size = 0;

    if (sub_block & 1) {
        neighbor = input_data + block_start;
        neighbor_size = 1 << (sorted_k - 1);
        offset = upper_bound(neighbor, neighbor_size, value);
    } else {
        neighbor = input_data + ((sub_block + 1) << (sorted_k - 1));
        if (neighbor < input_data + n) {
            int rem = (input_data + n) - neighbor;
            if (rem < (1 << (sorted_k - 1))) {
                neighbor_size = rem;
            } else {
                neighbor_size = (1 << (sorted_k - 1));
            }
        }
        offset = lower_bound(neighbor, neighbor_size, value);
    }

    output_data[block_start + offset + index_in_sub_block] = value;
}
