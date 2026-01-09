#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global uint* prefix_sum,
    unsigned int n,
    unsigned int level)
{
    unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    unsigned int block_size = 1u << level;
    unsigned int block_id = i / block_size;

    if ((block_id & 1) == 1) {
        unsigned int prev_block_last = block_id * block_size - 1;
        prefix_sum[i] += prefix_sum[prev_block_last];
    }
}
