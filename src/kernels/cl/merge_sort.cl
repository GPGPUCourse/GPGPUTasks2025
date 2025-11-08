#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint lower_bound(__global const uint* data, int size, uint value) {
    int lb = -1;
    int rb = size;
    while (lb + 1 < rb) {
        int mid = (lb + rb) / 2;
        if (data[mid] >= value) {
            rb = mid;
        } else {
            lb = mid;
        }
    }
    return rb;
}

uint upper_bound(__global const uint* data, int size, uint value) {
    int lb = -1;
    int rb = size;
    while (lb + 1 < rb) {
        int mid = (lb + rb) / 2;
        if (data[mid] > value) {
            rb = mid;
        } else {
            lb = mid;
        }
    }
    return rb;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    

    if (i >= n) {
        return;
    }

    uint value = input_data[i];
    uint block_index = i >> (k - 1);
    uint big_block_index = i >> k;

    uint other_index = 0;
    if (block_index & 1) {
        other_index = upper_bound(input_data + (big_block_index << k), 1 << (k - 1), value);
    } else {
        __global const uint* data = input_data + ((block_index + 1) << (k - 1));
        int size = 0;
        if (data < input_data + n) {
            int tmp = (input_data + n) - data;
            size = min(1 << (k - 1), tmp);
        }
        other_index = lower_bound(data, size, value);
    }

    output_data[(big_block_index << k) + other_index + (i - (block_index << (k - 1)))] = value;
}
