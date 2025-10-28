#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE_X, BUCKETS_COUNT, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* local_histograms,  // contains num_groups * BUCKETS_COUNT values
    __global       uint* reduced_histograms, //  will contain (num_groups/2) * BUCKETS_COUNT values
    unsigned int num_groups)
{
    const unsigned int bucket = get_global_id(1);
    const unsigned int i = get_global_id(0);
    const unsigned int local_id = get_local_id (0);
    if (i >= (num_groups + 1)/2) {
        return;
    }
    unsigned int idx0 = (i * 2) * BUCKETS_COUNT + bucket;
    unsigned int idx1 = (i * 2 + 1) * BUCKETS_COUNT + bucket;

    uint sum = local_histograms[idx0];
    if (i * 2 + 1 < num_groups) {
        sum += local_histograms[idx1];
    }
    reduced_histograms[i * BUCKETS_COUNT + bucket] = sum;
}
