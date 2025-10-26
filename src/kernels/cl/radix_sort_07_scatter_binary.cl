#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_07_scatter_binary(
    __global const uint* origInput,
    __global const uint* scanInput,
    __global       uint* output,
    unsigned int n,
    unsigned int bitsBlockIdx)
{
    unsigned int offset = scanInput[n];
    unsigned int idx = get_group_id(0) * SCATTER_BLOCK_SIZE * GROUP_SIZE + get_local_id(0);
    for (unsigned int i = 0; i < SCATTER_BLOCK_SIZE && idx < n; ++i) {
        uint val = origInput[idx];
        if (((val >> bitsBlockIdx) & 1) == 0) {
            output[scanInput[idx] - 1] = val;
        } else {
            output[idx - scanInput[idx] + offset] = val;
        }
        idx += GROUP_SIZE;
    }
}