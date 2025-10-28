#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h" 

__attribute__((reqd_work_group_size(GROUP_SIZE_X, BUCKETS_COUNT, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* reduced_histograms,  // num_groups/2 * BUCKETS_COUNT
    __global       uint* prefix_sum_accum,    // num_groups * BUCKETS_COUNT
    unsigned int num_groups,
    unsigned int pow2)
{
    const unsigned int i = get_global_id(0);
    const unsigned int bucket = get_global_id(1);

    if (i >= num_groups) {
        return;
    }

    const unsigned int idx = i * BUCKETS_COUNT + bucket;
    if (((i + 1) & (1 << pow2)) != 0) {
        uint offset = ((i + 1) >> pow2) - 1;
        prefix_sum_accum[i * BUCKETS_COUNT + bucket] += reduced_histograms[offset * BUCKETS_COUNT + bucket];
    }
}
