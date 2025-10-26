#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_05_global_prefixes_scan_sum_reduction_binary(
    __global const uint* input,
    __global       uint* output,
    unsigned int inputOffset,
    unsigned int outputOffset,
    unsigned int n,
    unsigned int bitsBlockIdx,
    unsigned int isFirstIter)
{
    unsigned int outIdx = get_global_id(0);
    unsigned int inIdx = outIdx * PREFIX_BLOCK_SIZE;
    if (inIdx + PREFIX_BLOCK_SIZE > n) {
        return;
    }
    input += inputOffset + inIdx;
    output += outputOffset + outIdx;

    unsigned int sum = 0;
    if (isFirstIter) {
        for (unsigned int i = 0; i < PREFIX_BLOCK_SIZE; ++i) {
            sum += ((input[i] >> bitsBlockIdx) & 1) == 0;
        }
    } else {
        for (unsigned int i = 0; i < PREFIX_BLOCK_SIZE; ++i) {
            sum += input[i];
        }
    }
    *output = sum;
}
