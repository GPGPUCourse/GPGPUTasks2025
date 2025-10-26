#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX_BUCKETS, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* block_histograms,
    __global uint* block_prefixes,
    __global uint* bucket_bases,
    unsigned int num_groups)
{
    const uint bucket = get_local_id(0);

    uint prefix = 0u;
    for (uint block = 0; block < num_groups; ++block) {
        const uint idx = block * RADIX_BUCKETS + bucket;
        const uint count = block_histograms[idx];
        block_prefixes[idx] = prefix;
        prefix += count;
    }

    __local uint totals[RADIX_BUCKETS];
    totals[bucket] = prefix;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint base = 0u;
    for (uint i = 0; i < bucket; ++i) {
        base += totals[i];
    }
    bucket_bases[bucket] = base;
}
