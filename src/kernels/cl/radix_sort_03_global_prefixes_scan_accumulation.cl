#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    const uint glob_id = get_global_id(0);
    if (glob_id >= a1)
        return;


    const uint bit = 1u << a2;
    if (((glob_id + 1) & bit) != 0) {
        const uint block_idx = (glob_id + 1) >> (a2 + 1);
        const uint left_block = block_idx << 1;
        const uint carry_base = left_block * 16;
        const uint out_base = glob_id * 16;

        for (int k = 0; k < 16; ++k) {
            buffer2[out_base + k] += buffer1[carry_base + k];
        }
    }

}
