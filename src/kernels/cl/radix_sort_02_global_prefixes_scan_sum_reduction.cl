#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global       uint* scratch,
    unsigned int n,
    unsigned int groups)
{
    if (get_global_id(0) != 0)
        return;

    const uint group_counts_base = n;
    const uint group_prefix_base = n + groups;

    uint prefix = 0;
    for (uint g = 0; g < groups; ++g) {
        const uint c = scratch[group_counts_base + g];
        scratch[group_prefix_base + g] = prefix;
        prefix += c;
    }
    scratch[group_prefix_base + groups] = 0; // будет заполнено в kernel3
}
