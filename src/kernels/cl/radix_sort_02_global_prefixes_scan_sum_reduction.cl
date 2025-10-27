#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1)
{
    const uint glob_id = get_global_id(0);
    const uint out_entries = (a1 + 1u) / 2;
    if (glob_id >= out_entries)
        return;
    const uint t0 = glob_id * 2;
    const uint t1 = t0 + 1;
    const uint base0 = t0 * 16;
    const uint base1 = t1 * 16;
    const uint outb = glob_id * 16;
    for (int k = 0; k < 16; ++k) {
        uint v0 = buffer1[base0 + k];
        uint v1 = (t1 < a1) ? buffer1[base1 + k] : 0;
        buffer2[outb + k] = v0 + v1;
    }
}
