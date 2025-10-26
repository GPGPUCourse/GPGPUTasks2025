#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_06_global_prefixes_scan_accumulation_binary(
    __global const uint* origInput,
    __global const uint* reductInput,
    __global       uint* output,
    unsigned int n,
    unsigned int bitsBlockIdx)
{
    unsigned int outIdx = get_global_id(0);

    unsigned int sum = 0;
    for (unsigned int i = outIdx / PREFIX_BLOCK_SIZE * PREFIX_BLOCK_SIZE; i <= outIdx; ++i) {
        sum += ((origInput[i] >> bitsBlockIdx) & 1) == 0;
    }
    unsigned int size = n / PREFIX_BLOCK_SIZE;
    unsigned int multiplier = PREFIX_BLOCK_SIZE;
    while (outIdx >= multiplier) {
        unsigned int curReduction = outIdx / multiplier;
        if (curReduction % PREFIX_BLOCK_SIZE != 0) {
            for (unsigned int inIndex = curReduction / PREFIX_BLOCK_SIZE * PREFIX_BLOCK_SIZE; inIndex < curReduction; ++inIndex) {
                sum += reductInput[inIndex];
            }
        }
        reductInput += size;
        size /= PREFIX_BLOCK_SIZE;
        multiplier *= PREFIX_BLOCK_SIZE;
    }
    if (outIdx < n) {
        output[outIdx] = sum;
        if (outIdx == n - 1) {
            output[n] = sum;
        }
    }
}
