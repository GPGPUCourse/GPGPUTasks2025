#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* counts,   // [numGroups * 2]: {zeros[g], ones[g]}
    __global       uint* offsets,  // [numGroups * 2]: {zeroOffset[g], oneOffset[g]} -> mutate oneOffset by adding totalZeros
    unsigned int numGroups,
    unsigned int unused)
{
    uint totalZeros = 0u;
    for (uint g = 0u; g < numGroups; ++g) {
        totalZeros += counts[g * 2 + 0];
    }

    for (uint g = 0u; g < numGroups; ++g) {
        offsets[g * 2 + 1] += totalZeros;
    }
}