#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* input,
    __global       uint* output,
    unsigned int n,
    unsigned int pow2)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    if (((global_id + 1) & (1 << pow2)) > 0) {
        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int idx = ((global_id + 1) >> pow2) - 1;
            output[global_id * BLOCK_SIZE + i] += input[idx * BLOCK_SIZE + i];
        }
    }
}
