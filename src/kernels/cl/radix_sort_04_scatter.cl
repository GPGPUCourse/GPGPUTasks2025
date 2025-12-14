#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sum,
    __global       uint* output,
    unsigned int n,
    unsigned int bit_number)
{
    unsigned int i = get_global_id(0);
    if (i < n) {
        bool is_bit_zero = (input[i] & (1 << bit_number)) == 0;
        unsigned int res_idx = is_bit_zero ? prefix_sum[i] - 1 : prefix_sum[n - 1] + (i - prefix_sum[i]);

        output[res_idx] = input[i];
    }
}