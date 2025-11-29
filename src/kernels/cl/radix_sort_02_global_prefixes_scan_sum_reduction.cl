#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts,
    __global uint* offsets,
    unsigned int numGroups)
{
    uint sums[16];
    for (uint i = 0; i < 16; ++i) {
        sums[i] = 0;
    }

    // Exclusive scan across groups for each bucket
    for (uint g = 0u; g < numGroups; ++g) {
        for (uint i = 0; i < 16; ++i) {
            offsets[g * 16 + i] = sums[i];
            sums[i] += counts[g * 16 + i];
        }
    }
}
