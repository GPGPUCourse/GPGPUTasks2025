#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* bins,
    __global const uint* scatter,
    __global const uint* global_scan,
    __global uint* output,
    const uint compressed,
    const uint n)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }

#pragma unroll
    for (uint bin = 0, global_offset = 0; bin < BIN_COUNT; global_offset += global_scan[(++bin) * compressed - 1]) {
        if (bins[index] == bin) {
            output[scatter[index] + (index >= GROUP_SIZE ? global_scan[bin * compressed + (index / GROUP_SIZE) - 1] : 0) + global_offset] = input[index];
        }
    }
}
