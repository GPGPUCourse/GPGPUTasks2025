#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global       uint* output,
    uint offset)
{
    const uint glid = get_global_id(0);
    const uint groups = get_num_groups(0);

    if (glid >= groups) return;

    uint val = input[glid];
    if (glid >= offset) val += input[glid - offset];
    output[glid] = val;
}
