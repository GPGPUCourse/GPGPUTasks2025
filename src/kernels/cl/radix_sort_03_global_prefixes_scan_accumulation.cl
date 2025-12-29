#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global       uint* scratch,
    unsigned int n,
    unsigned int groups)
{
    if (get_global_id(0) != 0)
        return;

    const uint group_counts_base = n;
    const uint group_prefix_base = n + groups;

    if (groups == 0) {
        scratch[group_prefix_base + groups] = 0;
        return;
    }

    const uint last_prefix = scratch[group_prefix_base + (groups - 1)];
    const uint last_count = scratch[group_counts_base + (groups - 1)];
    scratch[group_prefix_base + groups] = last_prefix + last_count;
}
