#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts,   // [numGroups * 2]: {zeros[g], ones[g]}
    __global       uint* offsets,  // [numGroups * 2]: {zeroOffset[g], oneOffset[g]}
    unsigned int numGroups)
{
    uint sumZeros = 0u;
    uint sumOnes  = 0u;
    for (uint g = 0u; g < numGroups; ++g) {
        uint z = counts[g * 2 + 0];
        uint o = counts[g * 2 + 1];
        offsets[g * 2 + 0] = sumZeros; // zero base for group g
        offsets[g * 2 + 1] = sumOnes;  // one  base for group g (will be shifted by totalZeros in kernel 03)
        sumZeros += z;
        sumOnes  += o;
    }
}
