#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* in_sums,
    __global       uint* out_sums,
    const uint m)
{
    const uint i  = get_global_id(0);
    const uint i2 = i << 1;
    if (i2 + 1 < m)
        out_sums[i] = in_sums[i2] + in_sums[i2 + 1];
    else if (i2 < m)
        out_sums[i] = in_sums[i2];
}
