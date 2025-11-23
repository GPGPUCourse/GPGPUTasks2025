#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* local_counts,
    __global const uint* global_sums,
    __global       uint* local_prefix_sums,
    unsigned int num_groups)
{
    const unsigned int bucket = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    
    __local uint global_prefix[RADIX_BUCKET_COUNT];
    if (bucket == 0) {
        global_prefix[0] = 0;
        for (unsigned int i = 1; i < RADIX_BUCKET_COUNT; ++i) {
            global_prefix[i] = global_prefix[i - 1] + global_sums[i - 1];
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (local_id < RADIX_BUCKET_COUNT) {
        uint prefix = global_prefix[local_id];
        for (unsigned int group = 0; group < num_groups; ++group) {
            local_prefix_sums[group * RADIX_BUCKET_COUNT + local_id] = prefix;
            prefix += local_counts[group * RADIX_BUCKET_COUNT + local_id];
        }
    }
}
