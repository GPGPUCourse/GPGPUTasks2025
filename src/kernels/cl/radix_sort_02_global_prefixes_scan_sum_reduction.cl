#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global uint* output,
    unsigned int n)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);

    const uint base0 = gid * 2;
    const uint base1 = gid * 2 + 1;

    const uint elm0 = base0 < n ? input[base0 * 16 + lid] : 0;
    const uint elm1 = base1 < n ? input[base1 * 16 + lid] : 0;

    output[gid * 16 + lid] = elm0 + elm1;
}
