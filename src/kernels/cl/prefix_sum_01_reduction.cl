#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global       uint* output,
    unsigned int inputOffset,
    unsigned int outputOffset,
    unsigned int n)
{
    unsigned int outIdx = get_global_id(0);
    unsigned int inIdx = outIdx * BLOCK_SIZE;
    if (inIdx + BLOCK_SIZE > n) {
        return;
    }
    input += inputOffset + inIdx;
    output += outputOffset + outIdx;

    unsigned int sum = 0;
    for (unsigned int i = 0; i < BLOCK_SIZE; ++i) {
        sum += input[i];
    }
    *output = sum;
}
