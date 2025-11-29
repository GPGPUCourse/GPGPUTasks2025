#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* counts,
    __global uint* offsets,
    unsigned int numGroups,
    unsigned int unused)
{
    uint total_counts[16];
    for (uint i = 0; i < 16; ++i) {
        total_counts[i] = 0;
    }

    // Calculate total counts per bucket
    for (uint g = 0u; g < numGroups; ++g) {
        for (uint i = 0; i < 16; ++i) {
            total_counts[i] += counts[g * 16 + i];
        }
    }

    // Calculate global offsets for each bucket
    uint prefix_sum = 0;
    for (uint i = 0; i < 16; ++i) {
        uint count = total_counts[i];
        total_counts[i] = prefix_sum; // Reuse array to store global offset
        prefix_sum += count;
    }

    // Add global bucket offsets
    for (uint g = 0u; g < numGroups; ++g) {
        for (uint i = 0; i < 16; ++i) {
            offsets[g * 16 + i] += total_counts[i];
        }
    }
}

