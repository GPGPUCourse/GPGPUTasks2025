#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* ones_per_group,
    __global       uint* prefix_groups,
    const uint m)
{
    const uint lid = get_local_id(0);
    if (lid == 0) {
        uint acc = 0u;
        for (uint g = 0; g < m; ++g) {
            acc += ones_per_group[g];
            prefix_groups[g] = acc;
        }
    }
}
