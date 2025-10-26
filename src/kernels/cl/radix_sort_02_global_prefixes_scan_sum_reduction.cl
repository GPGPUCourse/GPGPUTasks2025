#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global       uint* output,
    unsigned int n)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= (n + 1) / 2) {
        return;
    }

    unsigned int group_id = get_group_id(0);
    for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
        unsigned int n1 = input[2 * global_id * BLOCK_SIZE + i];
        unsigned int idx = (2 * global_id + 1) * BLOCK_SIZE + i;
        unsigned int n2 = 0;
        if (idx < n * BLOCK_SIZE) {
            n2 = input[idx];
        }

        output[global_id * BLOCK_SIZE + i] = n1 + n2;
    }
}
