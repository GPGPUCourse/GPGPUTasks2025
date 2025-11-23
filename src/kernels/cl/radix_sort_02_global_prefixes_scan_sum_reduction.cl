#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* local_counts,
    __global       uint* global_sums,
    unsigned int num_groups)
{
    const unsigned int bucket = get_global_id(0);
    
    if (bucket < RADIX_BUCKET_COUNT) {
        uint sum = 0;
        for (unsigned int group = 0; group < num_groups; ++group) {
            sum += local_counts[group * RADIX_BUCKET_COUNT + bucket];
        }
        global_sums[bucket] = sum;
    }
}
