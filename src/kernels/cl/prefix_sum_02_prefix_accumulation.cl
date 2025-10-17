#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* origInput,
    __global const uint* reductInput,
    __global       uint* output,
    unsigned int n)
{
    unsigned int outIdx = get_global_id(0);

    unsigned int sum = 0;
    for (unsigned int i = outIdx / BLOCK_SIZE * BLOCK_SIZE; i <= outIdx; ++i) {
        sum += origInput[i];
    }
    unsigned int size = n / BLOCK_SIZE;
    unsigned int multiplier = BLOCK_SIZE;
    while (outIdx >= multiplier) {
        unsigned int curReduction = outIdx / multiplier;
        if (curReduction % BLOCK_SIZE != 0) {
            for (unsigned int inIndex = curReduction / BLOCK_SIZE * BLOCK_SIZE; inIndex < curReduction; ++inIndex) {
                sum += reductInput[inIndex];
            }
        }
        reductInput += size;
        size /= BLOCK_SIZE;
        multiplier *= BLOCK_SIZE;
    }
    if (outIdx < n) {
        output[outIdx] = sum;
    }
}
